"""Training loops for all CRISPR-MTL experiments.

Single-task: A1, A2, B1, B2
Multi-task:  MTL-Full, ABL1, ABL2, ABL3
"""

import copy
import logging
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluate import compute_metrics_offtarget, compute_metrics_ontarget
from model import (
    BiLSTMBaseline,
    CNNBiLSTMBaseline,
    CRISPRMultiTask,
    DNABERTSingleTask,
)

logger = logging.getLogger(__name__)

# ─── Output directories ────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_CKPT_DIR  = _ROOT / "outputs" / "checkpoints"
_RESULTS_DIR = _ROOT / "outputs" / "results"

for _d in (_CKPT_DIR, _RESULTS_DIR, _ROOT / "outputs" / "figures"):
    _d.mkdir(parents=True, exist_ok=True)


# ─── DNA encoding utilities (for baseline models) ─────────────────────────────

_NUC_IDX  = {"A": 0, "C": 1, "G": 2, "T": 3}
_PURINES   = frozenset("AG")
_PYRIMIDINES = frozenset("CT")


def _encode_onehot(seq: str) -> torch.Tensor:
    """One-hot encode a DNA sequence. Returns (len, 4)."""
    out = torch.zeros(len(seq), 4)
    for i, nt in enumerate(seq.upper()):
        if nt in _NUC_IDX:
            out[i, _NUC_IDX[nt]] = 1.0
    return out


def _encode_mismatch(grna: str, dna: str) -> torch.Tensor:
    """Encode gRNA/DNA pair as 7-channel mismatch matrix. Returns (min_len, 7).

    ch 0-3 : gRNA one-hot (A, C, G, T)
    ch 4   : exact match at position
    ch 5   : transition mismatch (A↔G or C↔T)
    ch 6   : transversion mismatch (purine↔pyrimidine)
    """
    n = min(len(grna), len(dna))
    out = torch.zeros(n, 7)
    for i in range(n):
        g, d = grna[i].upper(), dna[i].upper()
        if g in _NUC_IDX:
            out[i, _NUC_IDX[g]] = 1.0
        if g == d:
            out[i, 4] = 1.0
        elif (g in _PURINES) == (d in _PURINES):
            out[i, 5] = 1.0   # transition
        else:
            out[i, 6] = 1.0   # transversion
    return out


# ─── Baseline datasets ─────────────────────────────────────────────────────────

class _OnTargetBaselineDataset(Dataset):
    """One-hot encoded on-target dataset for BiLSTMBaseline (A1)."""

    def __init__(self, df: pd.DataFrame) -> None:
        df = df.reset_index(drop=True)
        self.X = torch.stack([_encode_onehot(s) for s in df["grna_23mer"]])
        self.y = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict:
        return {"x": self.X[idx], "label": self.y[idx]}


class _OffTargetBaselineDataset(Dataset):
    """Mismatch-matrix encoded off-target dataset for CNNBiLSTMBaseline (A2)."""

    def __init__(self, df: pd.DataFrame) -> None:
        df = df.reset_index(drop=True)
        self.X = torch.stack([_encode_mismatch(g, d) for g, d in zip(df["grna"], df["dna_target"])])
        self.y = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict:
        return {"x": self.X[idx], "label": self.y[idx]}


# ─── Loss helpers ─────────────────────────────────────────────────────────────

def _weighted_bce(pred: torch.Tensor, target: torch.Tensor, w_pos: float) -> torch.Tensor:
    """Binary cross-entropy with per-class weighting (equivalent to BCELoss pos_weight).

    pred   : (batch,) in [0, 1] (post-sigmoid)
    target : (batch,) binary float
    w_pos  : weight for positive class  (= n_neg / n_pos, capped at 50)
    """
    weights = target * (w_pos - 1.0) + 1.0   # w_pos for positives, 1.0 for negatives
    return F.binary_cross_entropy(pred, target, weight=weights)


# ─── Forward helpers ──────────────────────────────────────────────────────────

def _forward(model: nn.Module, batch: Dict, task: Optional[str] = None) -> torch.Tensor:
    """Unified forward pass for baseline and BERT models."""
    if "x" in batch:
        return model(batch["x"])
    if isinstance(model, CRISPRMultiTask):
        return model(
            batch["input_ids"], batch["attention_mask"],
            task=task,
            token_type_ids=batch.get("token_type_ids"),
        )
    return model(
        batch["input_ids"], batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
    )


def _to_device(batch: Dict, device: torch.device) -> Dict:
    """Move all tensor values in a batch dict to device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# ─── Core training / eval loops ───────────────────────────────────────────────

def train_epoch_single(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    loss_fn,
    device: torch.device,
    task: Optional[str] = None,
) -> float:
    """Train one epoch for a single-task model. Returns mean loss."""
    model.train()
    losses: List[float] = []
    for batch in tqdm(loader, desc="  train", leave=False):
        batch = _to_device(batch, device)
        pred  = _forward(model, batch, task=task).squeeze(-1)
        loss  = loss_fn(pred, batch["label"])
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> Dict:
    """Evaluate model on a DataLoader. Returns metric dict."""
    model.eval()
    all_preds:  List[float] = []
    all_labels: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch  = _to_device(batch, device)
            pred   = _forward(model, batch, task=task).squeeze(-1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(batch["label"].cpu().numpy().tolist())
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    if task == "ontarget":
        return compute_metrics_ontarget(y_true, y_pred)
    return compute_metrics_offtarget(y_true, y_pred)


# ─── Optimizer builders ────────────────────────────────────────────────────────

def _build_optimizer_phase1(model: nn.Module, lr: float, wd: float) -> AdamW:
    """Phase-1 optimizer: only non-BERT (frozen-BERT) trainable params."""
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=lr, weight_decay=wd)


def _build_optimizer_phase2(model: nn.Module, lr_head: float, lr_bert: float, wd: float) -> AdamW:
    """Phase-2 optimizer: discriminative LR — BERT unfrozen layers vs heads."""
    bert_trainable = [p for p in model.bert.parameters() if p.requires_grad]
    bert_ids       = set(id(p) for p in bert_trainable)
    head_params    = [p for p in model.parameters() if p.requires_grad and id(p) not in bert_ids]
    groups = []
    if bert_trainable:
        groups.append({"params": bert_trainable, "lr": lr_bert})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})
    return AdamW(groups, weight_decay=wd)


# ─── CSV result logging ───────────────────────────────────────────────────────

def _append_result(row: Dict, csv_path: Path) -> None:
    """Append one result row to CSV. Creates file with header if missing."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode="w", header=True, index=False)


def _combined_score(metrics: Dict) -> float:
    """Compute combined score for checkpoint selection.

    0.5*(spearman+auroc) when both present; otherwise use the present metric.
    """
    s = metrics.get("spearman", None)
    a = metrics.get("auroc",    None)
    if s is not None and a is not None:
        return 0.5 * (s + a)
    return s if s is not None else (a if a is not None else 0.0)


# ─── Data loading helpers ─────────────────────────────────────────────────────

def _load_all_data(config: dict) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Load on-target df, off-target meta dict, and CV splits from config paths."""
    from dataset import load_config, load_offtarget, load_ontarget

    root = Path(__file__).parent.parent
    df_on     = load_ontarget(root / config["paths"]["ontarget_csv"])
    off_meta  = load_offtarget(
        root / config["paths"]["offtarget_dc"],
        root / config["paths"]["offtarget_lg"],
    )
    splits_path = root / config["paths"]["cv_splits"]
    if not splits_path.exists():
        from dataset import create_cv_splits
        splits = create_cv_splits(
            df_on,
            off_meta["df"],
            n_splits=config["training"]["n_folds"],
            seed=config["training"]["seed"],
            save_path=splits_path,
        )
    else:
        with open(splits_path, "rb") as f:
            splits = pkl.load(f)
    return df_on, off_meta, splits


# ─── train_single_task ────────────────────────────────────────────────────────

def train_single_task(
    exp_id: str,
    config: dict,
    device: torch.device,
    # ── test hooks (do not use in production) ──
    _test_loaders: Optional[List[Tuple[DataLoader, DataLoader]]] = None,
    _test_bert: Optional[nn.Module] = None,
) -> None:
    """Run 5-fold CV for single-task experiments A1, A2, B1, B2.

    _test_loaders : list of (train_loader, val_loader) per fold (bypasses file I/O)
    _test_bert    : pre-built BertModel injected into B1/B2 (bypasses HF download)
    """
    assert exp_id in ("A1", "A2", "B1", "B2"), f"Unknown exp_id for single-task: {exp_id}"

    task     = "ontarget"  if exp_id in ("A1", "B1") else "offtarget"
    is_bert  = exp_id in ("B1", "B2")
    is_regression = task == "ontarget"

    csv_path = _ROOT / config["paths"]["results_csv"]
    ckpt_dir = _ROOT / config["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data (skipped if _test_loaders provided) ──
    tokenizer = None
    if _test_loaders is None:
        df_on, off_meta, splits = _load_all_data(config)
        if is_bert:
            from transformers import BertTokenizer
            logger.info("Loading tokenizer: %s", config["model"]["dnabert_name"])
            tokenizer = BertTokenizer.from_pretrained(config["model"]["dnabert_name"])
    else:
        splits = {i: None for i in range(len(_test_loaders))}

    # ── Choose epoch/lr config ──
    cfg_key = "baseline" if exp_id in ("A1", "A2") else "dnabert_single"
    epochs_total  = config[cfg_key]["epochs"] if exp_id in ("A1", "A2") else config[cfg_key]["epochs_total"]
    warmup_epochs = 0 if exp_id in ("A1", "A2") else config[cfg_key]["warmup_epochs"]
    lr_head       = config[cfg_key]["lr"] if exp_id in ("A1", "A2") else config[cfg_key]["lr_head"]
    lr_bert       = config["dnabert_single"]["lr_dnabert"] if is_bert else None
    wd            = config["training"]["weight_decay"]
    w_pos         = min(off_meta["w_pos"], 50.0) if _test_loaders is None and not is_regression else None

    all_metrics: List[Dict] = []
    n_folds = len(splits)

    for fold in range(n_folds):
        ckpt_path = ckpt_dir / f"{exp_id}_fold{fold}_best.pt"
        if ckpt_path.exists():
            logger.info("Fold %d: checkpoint exists, skipping.", fold)
            continue

        logger.info("=== %s | Fold %d/%d ===", exp_id, fold, n_folds - 1)

        # ── Build DataLoaders ──
        if _test_loaders is not None:
            train_loader, val_loader = _test_loaders[fold]
        else:
            from dataset import (
                CRISPROffTargetDataset,
                CRISPROnTargetDataset,
            )
            if exp_id == "A1":
                train_loader = DataLoader(
                    _OnTargetBaselineDataset(df_on.iloc[splits[fold]["train_on"]]),
                    batch_size=config["training"]["batch_size_ontarget"], shuffle=True,
                )
                val_loader = DataLoader(
                    _OnTargetBaselineDataset(df_on.iloc[splits[fold]["val_on"]]),
                    batch_size=config["training"]["batch_size_ontarget"],
                )
            elif exp_id == "A2":
                df_off = off_meta["df"]
                train_loader = DataLoader(
                    _OffTargetBaselineDataset(df_off.iloc[splits[fold]["train_off"]]),
                    batch_size=config["training"]["batch_size_offtarget"], shuffle=True,
                )
                val_loader = DataLoader(
                    _OffTargetBaselineDataset(df_off.iloc[splits[fold]["val_off"]]),
                    batch_size=config["training"]["batch_size_offtarget"],
                )
            elif exp_id == "B1":
                train_loader = DataLoader(
                    CRISPROnTargetDataset(df_on.iloc[splits[fold]["train_on"]].reset_index(drop=True), tokenizer, config),
                    batch_size=config["training"]["batch_size_ontarget"], shuffle=True,
                )
                val_loader = DataLoader(
                    CRISPROnTargetDataset(df_on.iloc[splits[fold]["val_on"]].reset_index(drop=True), tokenizer, config),
                    batch_size=config["training"]["batch_size_ontarget"],
                )
            else:  # B2
                df_off = off_meta["df"]
                train_loader = DataLoader(
                    CRISPROffTargetDataset(df_off.iloc[splits[fold]["train_off"]].reset_index(drop=True), tokenizer, config),
                    batch_size=config["training"]["batch_size_offtarget"], shuffle=True,
                )
                val_loader = DataLoader(
                    CRISPROffTargetDataset(df_off.iloc[splits[fold]["val_off"]].reset_index(drop=True), tokenizer, config),
                    batch_size=config["training"]["batch_size_offtarget"],
                )

        # ── Build model ──
        if exp_id == "A1":
            model = BiLSTMBaseline(config).to(device)
        elif exp_id == "A2":
            model = CNNBiLSTMBaseline(config).to(device)
        else:  # B1/B2
            bert  = copy.deepcopy(_test_bert) if _test_bert is not None else None
            model = DNABERTSingleTask(config, task=task, _bert=bert).to(device)

        # ── Loss function ──
        if is_regression:
            loss_fn = nn.MSELoss()
        else:
            _wpos = w_pos if w_pos is not None else 9.0  # fallback for test mode
            loss_fn = lambda p, t, w=_wpos: _weighted_bce(p, t, w)

        # ── Phase 1 (warmup): all DNABERT frozen ──
        if is_bert:
            model.freeze_strategy("freeze_all")
        optimizer = _build_optimizer_phase1(model, lr_head, wd)

        best_val   = -np.inf
        best_epoch = 0
        patience   = config["training"]["early_stopping_patience"]
        no_improve = 0

        for epoch in range(1, epochs_total + 1):
            # Switch to Phase 2 after warmup
            if is_bert and epoch == warmup_epochs + 1:
                logger.info("  Phase 2: unfreezing DNABERT layers 9-12")
                model.freeze_strategy("freeze_8")
                optimizer = _build_optimizer_phase2(model, lr_head, lr_bert, wd)

            train_loss = train_epoch_single(model, train_loader, optimizer, loss_fn, device, task=task)
            val_metrics = eval_epoch(model, val_loader, device, task)
            combined = _combined_score(val_metrics)

            logger.info(
                "  Epoch %2d/%d | loss=%.4f | %s",
                epoch, epochs_total, train_loss,
                " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()),
            )

            if combined > best_val + config["training"]["min_delta"]:
                best_val   = combined
                best_epoch = epoch
                no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "fold": fold,
                        "exp_id": exp_id,
                        "model_state_dict": model.state_dict(),
                        "val_metrics": val_metrics,
                        "val_combined": combined,
                    },
                    ckpt_path,
                )
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("  Early stopping at epoch %d (best=%d)", epoch, best_epoch)
                    break

        # ── Load best checkpoint for final metrics ──
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            best_metrics = ckpt["val_metrics"]
        else:
            best_metrics = val_metrics  # fallback: last epoch

        row = {"exp_id": exp_id, "fold": fold}
        row.update({k: float("nan") for k in ("spearman", "pearson", "auroc", "aupr")})
        row.update(best_metrics)
        _append_result(row, csv_path)
        all_metrics.append(best_metrics)
        logger.info("  Fold %d best: %s", fold, best_metrics)

    # ── Summary ──
    if all_metrics:
        primary = "spearman" if task == "ontarget" else "auroc"
        vals = [m[primary] for m in all_metrics if primary in m]
        if vals:
            logger.info(
                "%s done | %s: %.3f ± %.3f",
                exp_id, primary, np.mean(vals), np.std(vals),
            )
            print(f"{exp_id} done | {primary}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")


# ─── train_multitask ──────────────────────────────────────────────────────────

def train_multitask(
    exp_id: str,
    config: dict,
    device: torch.device,
    # ── test hooks ──
    _test_loaders: Optional[List[Tuple]] = None,
    _test_bert: Optional[nn.Module] = None,
) -> None:
    """Run 5-fold CV for multi-task experiments: MTL-Full, ABL1, ABL2, ABL3.

    _test_loaders : list of (on_train, on_val, off_train, off_val) per fold
    _test_bert    : pre-built BertModel (bypasses HF download)
    """
    assert exp_id in ("MTL-Full", "ABL1", "ABL2", "ABL3"), \
        f"Unknown exp_id for multitask: {exp_id}"

    ablation = config.get("ablation", {})
    use_combined_loss = (exp_id == "ABL3")
    alpha    = ablation.get("abl3_alpha", 0.5) if use_combined_loss else None

    csv_path = _ROOT / config["paths"]["results_csv"]
    ckpt_dir = _ROOT / config["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg_mtl       = config["mtl"]
    epochs_total  = cfg_mtl["epochs_total"]
    warmup_epochs = cfg_mtl["warmup_epochs"]
    lr_head       = cfg_mtl["lr_head"]
    lr_bert       = cfg_mtl["lr_dnabert"]
    lr_bert_abl2  = ablation.get("abl2_lr_dnabert_early", 1e-6)
    wd            = config["training"]["weight_decay"]

    tokenizer = None
    if _test_loaders is None:
        df_on, off_meta, splits = _load_all_data(config)
        from transformers import BertTokenizer
        logger.info("Loading tokenizer: %s", config["model"]["dnabert_name"])
        tokenizer = BertTokenizer.from_pretrained(config["model"]["dnabert_name"])
        w_pos_raw = off_meta["w_pos"]
    else:
        splits = {i: None for i in range(len(_test_loaders))}
        w_pos_raw = 9.0  # dummy

    w_pos = min(w_pos_raw, 50.0)
    mse_fn = nn.MSELoss()

    all_on_metrics:  List[Dict] = []
    all_off_metrics: List[Dict] = []
    n_folds = len(splits)

    for fold in range(n_folds):
        ckpt_path = ckpt_dir / f"{exp_id}_fold{fold}_best.pt"
        if ckpt_path.exists():
            logger.info("Fold %d: checkpoint exists, skipping.", fold)
            continue

        logger.info("=== %s | Fold %d/%d ===", exp_id, fold, n_folds - 1)

        # ── Build DataLoaders ──
        if _test_loaders is not None:
            on_train, on_val, off_train, off_val = _test_loaders[fold]
        else:
            from dataset import CRISPROffTargetDataset, CRISPROnTargetDataset
            df_off = off_meta["df"]
            on_train = DataLoader(
                CRISPROnTargetDataset(df_on.iloc[splits[fold]["train_on"]].reset_index(drop=True), tokenizer, config),
                batch_size=config["training"]["batch_size_ontarget"], shuffle=True,
            )
            on_val = DataLoader(
                CRISPROnTargetDataset(df_on.iloc[splits[fold]["val_on"]].reset_index(drop=True), tokenizer, config),
                batch_size=config["training"]["batch_size_ontarget"],
            )
            off_train = DataLoader(
                CRISPROffTargetDataset(df_off.iloc[splits[fold]["train_off"]].reset_index(drop=True), tokenizer, config),
                batch_size=config["training"]["batch_size_offtarget"], shuffle=True,
            )
            off_val = DataLoader(
                CRISPROffTargetDataset(df_off.iloc[splits[fold]["val_off"]].reset_index(drop=True), tokenizer, config),
                batch_size=config["training"]["batch_size_offtarget"],
            )

        # ── Build model ──
        bert  = copy.deepcopy(_test_bert) if _test_bert is not None else None
        model = CRISPRMultiTask(config, _bert=bert).to(device)

        # ── Apply ABL1/ABL2 initial freeze ──
        freeze_init = ablation.get("abl1_freeze_strategy", "freeze_8") if exp_id == "ABL1" else "freeze_8"
        model.freeze_strategy(freeze_init)
        optimizer = _build_optimizer_phase1(model, lr_head, wd)

        best_combined = -np.inf
        best_epoch    = 0
        patience      = config["training"]["early_stopping_patience"]
        no_improve    = 0
        best_on_metrics  = {}
        best_off_metrics = {}

        for epoch in range(1, epochs_total + 1):
            # ── Phase 2 transition ──
            if epoch == warmup_epochs + 1:
                if exp_id == "ABL1":
                    pass  # stay frozen
                elif exp_id == "ABL2":
                    logger.info("  ABL2 Phase 2: unfreeze all DNABERT layers")
                    model.freeze_strategy("unfreeze_all")
                    optimizer = _build_optimizer_phase2(model, lr_head, lr_bert_abl2, wd)
                else:  # MTL-Full, ABL3
                    logger.info("  Phase 2: unfreezing DNABERT layers 9-12")
                    model.freeze_strategy("freeze_8")
                    optimizer = _build_optimizer_phase2(model, lr_head, lr_bert, wd)

            # ── Training epoch ──
            if use_combined_loss:
                ep_on_loss, ep_off_loss = _train_epoch_combined(
                    model, on_train, off_train, optimizer, mse_fn, w_pos, alpha, device,
                )
            else:
                ep_on_loss, ep_off_loss = _train_epoch_alternating(
                    model, on_train, off_train, optimizer, mse_fn, w_pos, device,
                )

            # ── Validation ──
            on_metrics  = eval_epoch(model, on_val,  device, "ontarget")
            off_metrics = eval_epoch(model, off_val, device, "offtarget")
            combined    = _combined_score({**on_metrics, **off_metrics})

            logger.info(
                "  Epoch %2d/%d | on_loss=%.4f off_loss=%.4f | spearman=%.4f auroc=%.4f",
                epoch, epochs_total, ep_on_loss, ep_off_loss,
                on_metrics.get("spearman", float("nan")),
                off_metrics.get("auroc",   float("nan")),
            )

            if combined > best_combined + config["training"]["min_delta"]:
                best_combined    = combined
                best_epoch       = epoch
                best_on_metrics  = on_metrics
                best_off_metrics = off_metrics
                no_improve       = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "fold": fold,
                        "exp_id": exp_id,
                        "model_state_dict": model.state_dict(),
                        "val_on_metrics":  on_metrics,
                        "val_off_metrics": off_metrics,
                        "val_combined":    combined,
                    },
                    ckpt_path,
                )
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("  Early stopping at epoch %d (best=%d)", epoch, best_epoch)
                    break

        # Use last epoch metrics if checkpoint wasn't saved (e.g., smoke test 2 epochs)
        if not best_on_metrics:
            best_on_metrics  = on_metrics
            best_off_metrics = off_metrics

        row = {"exp_id": exp_id, "fold": fold}
        row.update({k: float("nan") for k in ("spearman", "pearson", "auroc", "aupr")})
        row.update(best_on_metrics)
        row.update(best_off_metrics)
        _append_result(row, csv_path)
        all_on_metrics.append(best_on_metrics)
        all_off_metrics.append(best_off_metrics)
        logger.info("  Fold %d best: on=%s | off=%s", fold, best_on_metrics, best_off_metrics)

    # ── Summary ──
    if all_on_metrics:
        spear = [m.get("spearman", float("nan")) for m in all_on_metrics]
        auroc = [m.get("auroc",    float("nan")) for m in all_off_metrics]
        spear_v = [v for v in spear if not np.isnan(v)]
        auroc_v = [v for v in auroc if not np.isnan(v)]
        msg = f"{exp_id} done"
        if spear_v:
            msg += f" | Spearman: {np.mean(spear_v):.3f} ± {np.std(spear_v):.3f}"
        if auroc_v:
            msg += f" | AUROC: {np.mean(auroc_v):.3f} ± {np.std(auroc_v):.3f}"
        logger.info(msg)
        print(msg)


# ─── Alternating / combined epoch helpers ─────────────────────────────────────

def _train_epoch_alternating(
    model: CRISPRMultiTask,
    on_loader: DataLoader,
    off_loader: DataLoader,
    optimizer: AdamW,
    mse_fn: nn.MSELoss,
    w_pos: float,
    device: torch.device,
) -> Tuple[float, float]:
    """Alternating batch: one on-target step, one off-target step. Returns (on_loss, off_loss)."""
    model.train()
    on_losses: List[float] = []
    off_losses: List[float] = []
    on_iter  = iter(on_loader)
    off_iter = iter(off_loader)
    n_steps  = max(len(on_loader), len(off_loader))

    for _ in tqdm(range(n_steps), desc="  train", leave=False):
        # On-target step
        try:
            ob = next(on_iter)
        except StopIteration:
            on_iter = iter(on_loader)
            ob = next(on_iter)
        ob = _to_device(ob, device)
        on_pred = model(ob["input_ids"], ob["attention_mask"], task="ontarget",
                        token_type_ids=ob.get("token_type_ids")).squeeze(-1)
        on_loss = mse_fn(on_pred, ob["label"])
        optimizer.zero_grad()
        on_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        on_losses.append(on_loss.item())

        # Off-target step
        try:
            fb = next(off_iter)
        except StopIteration:
            off_iter = iter(off_loader)
            fb = next(off_iter)
        fb = _to_device(fb, device)
        off_pred = model(fb["input_ids"], fb["attention_mask"], task="offtarget",
                         token_type_ids=fb.get("token_type_ids")).squeeze(-1)
        off_loss = _weighted_bce(off_pred, fb["label"], w_pos)
        optimizer.zero_grad()
        off_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        off_losses.append(off_loss.item())

    return float(np.mean(on_losses)), float(np.mean(off_losses))


def _train_epoch_combined(
    model: CRISPRMultiTask,
    on_loader: DataLoader,
    off_loader: DataLoader,
    optimizer: AdamW,
    mse_fn: nn.MSELoss,
    w_pos: float,
    alpha: float,
    device: torch.device,
) -> Tuple[float, float]:
    """ABL3: combined loss L = alpha*L_on + (1-alpha)*L_off. Returns (on_loss, off_loss)."""
    model.train()
    on_losses: List[float] = []
    off_losses: List[float] = []
    on_iter  = iter(on_loader)
    off_iter = iter(off_loader)
    n_steps  = max(len(on_loader), len(off_loader))

    for _ in tqdm(range(n_steps), desc="  train", leave=False):
        try:
            ob = next(on_iter)
        except StopIteration:
            on_iter = iter(on_loader)
            ob = next(on_iter)
        try:
            fb = next(off_iter)
        except StopIteration:
            off_iter = iter(off_loader)
            fb = next(off_iter)

        ob = _to_device(ob, device)
        fb = _to_device(fb, device)

        on_pred  = model(ob["input_ids"], ob["attention_mask"], task="ontarget",
                         token_type_ids=ob.get("token_type_ids")).squeeze(-1)
        off_pred = model(fb["input_ids"], fb["attention_mask"], task="offtarget",
                         token_type_ids=fb.get("token_type_ids")).squeeze(-1)

        on_loss  = mse_fn(on_pred, ob["label"])
        off_loss = _weighted_bce(off_pred, fb["label"], w_pos)
        loss     = alpha * on_loss + (1.0 - alpha) * off_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        on_losses.append(on_loss.item())
        off_losses.append(off_loss.item())

    return float(np.mean(on_losses)), float(np.mean(off_losses))


# ─── run_experiment (top-level dispatcher) ────────────────────────────────────

def run_experiment(
    exp_id: str,
    config: dict,
    device: Optional[torch.device] = None,
) -> None:
    """Top-level dispatcher. Runs one full experiment with 5-fold CV.

    Calls train_single_task for A1/A2/B1/B2,
    calls train_multitask for MTL-Full/ABL1/ABL2/ABL3.
    Auto-detects device if not provided.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("run_experiment: %s on %s", exp_id, device)

    if exp_id in ("A1", "A2", "B1", "B2"):
        train_single_task(exp_id, config, device)
    elif exp_id in ("MTL-Full", "ABL1", "ABL2", "ABL3"):
        train_multitask(exp_id, config, device)
    else:
        raise ValueError(f"Unknown exp_id: {exp_id!r}")
