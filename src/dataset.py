"""Data loading, preprocessing, tokenization, and CV splits for CRISPR-MTL."""

import copy
import logging
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandas.compat.pickle_compat as pc
from sklearn.model_selection import KFold, StratifiedGroupKFold

# torch is imported lazily inside Dataset classes so this module can be loaded
# in environments where PyTorch DLLs may not be available (e.g. local Windows dev).
try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except OSError:
    _TORCH_AVAILABLE = False
    Dataset = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ─── Pandas 2.x compatibility patch for old BlockManager pickles ──────────────

def _patch_pickle_compat() -> None:
    """Patch pandas Unpickler for full compatibility with old pandas pickles.

    Handles two issues:
    1. BlockManager reconstruction fails on pandas >= 2.0 (TypeError)
    2. 'pandas.indexes.*' module paths no longer exist (moved to
       'pandas.core.indexes.*' in pandas 0.20+), causing ModuleNotFoundError.
    """
    from pandas.core.internals import BlockManager
    import importlib

    # ── Patch 1: load_reduce — fix BlockManager __new__ ──────────────────────
    def _load_reduce(self) -> None:
        stack = self.stack
        args = stack.pop()
        func = stack[-1]
        try:
            stack[-1] = func(*args)
        except TypeError as err:
            if "BlockManager" in str(err) and args and isinstance(args[0], type):
                cls = args[0]
                stack[-1] = cls.__new__(cls, (), [], False)
                return
            msg = "_reconstruct: First argument must be a sub-type of ndarray"
            if msg in str(err):
                try:
                    stack[-1] = object.__new__(args[0])
                    return
                except TypeError:
                    pass
            raise

    # ── Patch 2: find_class — remap old pandas.indexes.* module paths ─────────
    _orig_find_class = pc.Unpickler.find_class

    def _find_class(self, module: str, name: str):
        # pandas.indexes.* was renamed to pandas.core.indexes.* in pandas 0.20
        if module.startswith("pandas.indexes"):
            module = module.replace("pandas.indexes", "pandas.core.indexes", 1)
        # pandas.core.common.Index etc. — remap to core.indexes.base
        try:
            return _orig_find_class(self, module, name)
        except (ModuleNotFoundError, AttributeError, ImportError):
            # Fallback: search common pandas index locations
            for fallback_mod in (
                "pandas.core.indexes.base",
                "pandas.core.indexes.range",
                "pandas.core.indexes.multi",
                "pandas.core.frame",
                "pandas.core.series",
            ):
                try:
                    mod = importlib.import_module(fallback_mod)
                    if hasattr(mod, name):
                        return getattr(mod, name)
                except (ImportError, AttributeError):
                    continue
            # Re-raise original error if all fallbacks exhausted
            return _orig_find_class(self, module, name)

    pc.Unpickler.dispatch = copy.copy(pc.Unpickler.dispatch)
    pc.Unpickler.dispatch[pkl.REDUCE[0]] = _load_reduce
    pc.Unpickler.find_class = _find_class


_patch_pickle_compat()


# ─── Tokenization ──────────────────────────────────────────────────────────────

def seq_to_kmer(seq: str, k: int = 6) -> str:
    """Convert a DNA sequence to a space-separated k-mer string for DNABERT."""
    return " ".join([seq[i : i + k] for i in range(len(seq) - k + 1)])


# ─── Data Loading ──────────────────────────────────────────────────────────────

def load_ontarget(path: Path) -> pd.DataFrame:
    """Load on-target CSV. Extracts 23-mer from 30mer[4:27], clips label to [0, 1].

    Returns DataFrame with columns: grna_23mer, label.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"On-target file not found: {path}")

    df = pd.read_csv(path)
    missing = {"30mer", "predictions"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in on-target CSV: {missing}")

    df["grna_23mer"] = df["30mer"].str[4:27]
    df["label"] = np.clip(df["predictions"].astype(float), 0.0, 1.0)

    valid_mask = df["grna_23mer"].str.len() == 23
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        logger.warning("Dropped %d on-target rows with invalid 23-mer length", n_dropped)
    df = df[valid_mask].reset_index(drop=True)

    logger.info(
        "On-target loaded: %d samples | label range [%.4f, %.4f]",
        len(df), df["label"].min(), df["label"].max(),
    )
    return df[["grna_23mer", "label"]]


def load_offtarget(dc_path: Path, lg_path: Path) -> Dict:
    """Load and merge DeepCRISPR + Listgarten off-target datasets.

    Returns dict:
        'df'    : DataFrame with columns [grna, dna_target, label]
        'w_pos' : float = n_negative / n_positive  (pos_weight for BCE loss)
    """
    dc_path = Path(dc_path)
    lg_path = Path(lg_path)
    for p in (dc_path, lg_path):
        if not p.exists():
            raise FileNotFoundError(f"Off-target file not found: {p}")

    # --- DeepCRISPR: tab-separated, no header ---
    # col 0 = guide_id, col 1 = gRNA (23mer), col 6 = dna_target (23mer), col 11 = label
    dc_raw = pd.read_csv(dc_path, sep="\t", header=None)
    df_dc = pd.DataFrame({
        "grna":       dc_raw[1].astype(str),
        "dna_target": dc_raw[6].astype(str),
        "label":      dc_raw[11].astype(int),
    })
    logger.info(
        "DeepCRISPR loaded: %d rows | %d positive, %d negative",
        len(df_dc), int(df_dc["label"].sum()), int((df_dc["label"] == 0).sum()),
    )

    # --- Listgarten: pandas < 2.0 pickle, requires compat patch + latin-1 ---
    # col '30mer' = gRNA (23mer), '30mer_mut' = dna_target (23mer), 'wasValidated' = label
    # Use Unpickler directly — pc.load() is absent in some pandas versions.
    with open(lg_path, "rb") as f:
        lg_raw = pc.Unpickler(f, encoding="latin-1").load()
    df_lg = pd.DataFrame({
        "grna":       lg_raw["30mer"].astype(str),
        "dna_target": lg_raw["30mer_mut"].astype(str),
        "label":      lg_raw["wasValidated"].astype(int),
    })
    logger.info(
        "Listgarten loaded: %d rows | %d positive, %d negative",
        len(df_lg), int(df_lg["label"].sum()), int((df_lg["label"] == 0).sum()),
    )

    # --- Merge + deduplicate by (grna, dna_target) ---
    df = pd.concat([df_dc, df_lg], ignore_index=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=["grna", "dna_target"]).reset_index(drop=True)
    n_after = len(df)
    if n_before != n_after:
        logger.info("Deduplicated: %d → %d rows (removed %d duplicate pairs)",
                    n_before, n_after, n_before - n_after)

    n_pos = int(df["label"].sum())
    n_neg = int((df["label"] == 0).sum())
    w_pos = float(n_neg / n_pos)

    logger.info(
        "Off-target final: %d total | %d positive | %d negative | w_pos=%.2f",
        len(df), n_pos, n_neg, w_pos,
    )
    return {"df": df[["grna", "dna_target", "label"]], "w_pos": w_pos}


# ─── PyTorch Datasets ──────────────────────────────────────────────────────────

class CRISPROnTargetDataset(Dataset):
    """PyTorch Dataset for on-target efficiency prediction (regression)."""

    def __init__(self, df: pd.DataFrame, tokenizer, config: dict) -> None:
        seqs = [seq_to_kmer(s, k=config["data"]["kmer_k"]) for s in df["grna_23mer"]]
        enc = tokenizer(
            seqs,
            max_length=config["model"]["max_length_ontarget"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc.get("token_type_ids")
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        item: Dict = {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label":          self.labels[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item


class CRISPROffTargetDataset(Dataset):
    """PyTorch Dataset for off-target activity prediction (binary classification).

    Encodes gRNA + DNA target as a sentence pair:
    [CLS] gRNA_tokens [SEP] dna_tokens [SEP] (38 tokens total before padding).
    """

    def __init__(self, df: pd.DataFrame, tokenizer, config: dict) -> None:
        k = config["data"]["kmer_k"]
        grna_seqs = [seq_to_kmer(s, k=k) for s in df["grna"]]
        dna_seqs  = [seq_to_kmer(s, k=k) for s in df["dna_target"]]
        enc = tokenizer(
            grna_seqs,
            dna_seqs,
            max_length=config["model"]["max_length_offtarget"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc.get("token_type_ids")
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        item: Dict = {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label":          self.labels[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item


# ─── Cross-Validation Splits ──────────────────────────────────────────────────

def create_cv_splits(
    df_on: pd.DataFrame,
    df_off: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    save_path: Path = Path("data/processed/cv_splits.pkl"),
) -> Dict:
    """Create n-fold CV splits and save to disk.

    On-target: KFold (balanced classes).
    Off-target: StratifiedGroupKFold grouped by gRNA — prevents the same gRNA
    leaking across train/val (one gRNA spans many rows) while keeping the
    positive class balanced across folds (~53 positives total).

    Returns dict: fold_idx → {train_on, val_on, train_off, val_off} index lists.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    kf_on   = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sgkf_off = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    on_idx     = np.arange(len(df_on))
    off_idx    = np.arange(len(df_off))
    off_labels = df_off["label"].values
    off_groups = df_off["grna"].values   # group by gRNA → no leakage

    splits: Dict = {}
    for fold, ((tr_on, val_on), (tr_off, val_off)) in enumerate(
        zip(kf_on.split(on_idx),
            sgkf_off.split(off_idx, off_labels, groups=off_groups))
    ):
        splits[fold] = {
            "train_on":  on_idx[tr_on].tolist(),
            "val_on":    on_idx[val_on].tolist(),
            "train_off": off_idx[tr_off].tolist(),
            "val_off":   off_idx[val_off].tolist(),
        }
        n_pos_tr  = int(off_labels[tr_off].sum())
        n_pos_val = int(off_labels[val_off].sum())
        logger.info(
            "Fold %d | on-target: train=%d val=%d | "
            "off-target: train=%d (pos=%d) val=%d (pos=%d)",
            fold,
            len(tr_on), len(val_on),
            len(tr_off), n_pos_tr,
            len(val_off), n_pos_val,
        )

    with open(save_path, "wb") as f:
        pkl.dump(splits, f)
    logger.info("CV splits saved → %s", save_path)
    return splits


# ─── Config Loader ────────────────────────────────────────────────────────────

def load_config(path: Path = Path("configs/config.yaml")) -> dict:
    """Load YAML config from disk."""
    import yaml
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─── Smoke Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    root = Path(__file__).parent.parent
    config = load_config(root / "configs" / "config.yaml")

    df_on = load_ontarget(root / config["paths"]["ontarget_csv"])
    off   = load_offtarget(
        root / config["paths"]["offtarget_dc"],
        root / config["paths"]["offtarget_lg"],
    )
    df_off = off["df"]

    splits = create_cv_splits(
        df_on,
        df_off,
        n_splits=config["training"]["n_folds"],
        seed=config["training"]["seed"],
        save_path=root / config["paths"]["cv_splits"],
    )
    logger.info("Smoke test complete — %d folds, w_pos=%.2f", len(splits), off["w_pos"])
