"""Evaluation metrics, report generation, and interpretability for CRISPR-MTL."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Kaggle/Colab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


# ─── Metric computation ────────────────────────────────────────────────────────

def compute_metrics_ontarget(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute Spearman and Pearson correlation for on-target regression.

    Returns dict with keys: spearman, pearson.
    """
    spearman, _ = spearmanr(y_true, y_pred)
    pearson,  _ = pearsonr(y_true, y_pred)
    return {"spearman": float(spearman), "pearson": float(pearson)}


def compute_metrics_offtarget(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute AUROC and AUPR for off-target binary classification.

    Returns dict with keys: auroc, aupr.
    Falls back to 0.0 if only one class present (degenerate val fold).
    """
    try:
        auroc = float(roc_auc_score(y_true, y_pred))
        aupr  = float(average_precision_score(y_true, y_pred))
    except ValueError as e:
        logger.warning("Metric computation failed (likely single class in fold): %s", e)
        auroc = 0.0
        aupr  = 0.0
    return {"auroc": auroc, "aupr": aupr}


# ─── Results report ────────────────────────────────────────────────────────────

def generate_report(results_csv: Optional[Path] = None, save_csv: Optional[Path] = None) -> pd.DataFrame:
    """Load per-fold results, compute mean ± std per experiment, print and save.

    Args:
        results_csv: Path to per-fold results CSV. Defaults to outputs/results/results_table.csv.
        save_csv:    Path to save summary. Defaults to outputs/results/summary_table.csv.

    Returns:
        Summary DataFrame.
    """
    root = Path(__file__).parent.parent
    if results_csv is None:
        results_csv = root / "outputs" / "results" / "results_table.csv"
    if save_csv is None:
        save_csv = root / "outputs" / "results" / "summary_table.csv"

    if not Path(results_csv).exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    df = pd.read_csv(results_csv)

    metric_cols = [c for c in df.columns if c not in ("exp_id", "fold")]
    rows = []
    exp_order = ["A1", "A2", "B1", "B2", "MTL-Full", "ABL1", "ABL2", "ABL3"]

    for exp in exp_order:
        sub = df[df["exp_id"] == exp]
        if sub.empty:
            continue
        row = {"exp_id": exp, "n_folds": len(sub)}
        for col in metric_cols:
            vals = sub[col].dropna()
            if vals.empty:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"]  = float("nan")
            else:
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_std"]  = vals.std()
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Pretty print
    print("\n" + "=" * 72)
    print("CRISPR-MTL Results Summary")
    print("=" * 72)
    for _, r in summary.iterrows():
        print(f"\n{r['exp_id']}  ({int(r['n_folds'])} folds)")
        for col in metric_cols:
            m = r.get(f"{col}_mean", float("nan"))
            s = r.get(f"{col}_std", float("nan"))
            if not np.isnan(m):
                print(f"  {col:12s}: {m:.4f} ± {s:.4f}")
            else:
                print(f"  {col:12s}: N/A")
    print("=" * 72 + "\n")

    save_csv = Path(save_csv)
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(save_csv, index=False)
    logger.info("Summary saved → %s", save_csv)
    return summary


# ─── Integrated Gradients ──────────────────────────────────────────────────────

def compute_integrated_gradients(
    model,
    tokenizer,
    samples_on: pd.DataFrame,
    samples_off: pd.DataFrame,
    n_steps: int = 50,
    device: str = "cpu",
) -> Dict:
    """Compute per-nucleotide importance via Integrated Gradients on DNABERT embeddings.

    Uses captum.attr.IntegratedGradients on the word embedding layer.
    Aggregates k-mer token attributions back to nucleotide positions
    (sum absolute IG across overlapping 6-mer tokens, then L1-normalize).

    Args:
        model:       CRISPRMultiTask (eval mode)
        tokenizer:   DNABERT BertTokenizer
        samples_on:  DataFrame with column 'grna_23mer' (on-target samples)
        samples_off: DataFrame with column 'grna' and 'dna_target' (off-target)
        n_steps:     Riemann integration steps
        device:      'cpu' or 'cuda'

    Returns:
        dict with:
            'head1_importance': np.array shape (23,) — per-nucleotide importance for Head 1
            'head2_importance': np.array shape (23,) — per-nucleotide importance for Head 2
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        raise ImportError("captum is required for IG analysis. pip install captum")

    import torch
    from src.dataset import seq_to_kmer

    model = model.to(device)
    model.eval()

    def _get_embedding_fn(task: str):
        """Return a forward function that takes embeddings as input."""
        def forward_from_embeddings(embeddings: "torch.Tensor") -> "torch.Tensor":
            # embeddings: (batch, seq_len, hidden)
            # Bypass the embedding layer, feed directly into encoder
            attention_mask = torch.ones(
                embeddings.shape[0], embeddings.shape[1],
                dtype=torch.long, device=device
            )
            enc_out = model.bert.encoder(
                embeddings,
                attention_mask=model.bert.get_extended_attention_mask(
                    attention_mask, embeddings.shape[:2]
                ),
            )
            cls = enc_out.last_hidden_state[:, 0, :]
            x = model.projection(cls)
            return model.head_ontarget(x) if task == "ontarget" else model.head_offtarget(x)
        return forward_from_embeddings

    def _run_ig(seqs_a: List[str], seqs_b: Optional[List[str]], task: str) -> np.ndarray:
        """Run IG and return per-position attribution array of shape (23,)."""
        k = 6
        if seqs_b is None:
            enc = tokenizer(
                [seq_to_kmer(s, k) for s in seqs_a],
                padding="max_length", max_length=30,
                truncation=True, return_tensors="pt"
            )
        else:
            enc = tokenizer(
                [seq_to_kmer(s, k) for s in seqs_a],
                [seq_to_kmer(s, k) for s in seqs_b],
                padding="max_length", max_length=50,
                truncation=True, return_tensors="pt"
            )
        input_ids = enc["input_ids"].to(device)

        embeddings = model.bert.embeddings.word_embeddings(input_ids)
        baseline   = torch.zeros_like(embeddings)

        ig = IntegratedGradients(_get_embedding_fn(task))
        attributions = ig.attribute(
            embeddings,
            baselines=baseline,
            n_steps=n_steps,
            return_convergence_delta=False,
        )  # (batch, seq_len, hidden)

        # Aggregate: sum abs attributions over hidden dim → (batch, seq_len)
        token_attr = attributions.abs().sum(-1).detach().cpu().numpy()  # (batch, seq_len)
        mean_attr  = token_attr.mean(0)  # (seq_len,)

        # Map token positions back to nucleotide positions
        # For on-target: [CLS]=0, tokens 1..18 are k-mers for positions 0..17
        # Nucleotide i covered by k-mers max(0, i-k+1)..min(i, 17) → tokens +1 offset
        n_nuc = 23
        nuc_attr = np.zeros(n_nuc)
        for t_idx in range(1, 1 + (n_nuc - k + 1)):  # tokens 1..18
            nuc_start = t_idx - 1
            nuc_end   = nuc_start + k
            for nuc_pos in range(nuc_start, min(nuc_end, n_nuc)):
                nuc_attr[nuc_pos] += mean_attr[t_idx]

        # L1 normalize
        total = nuc_attr.sum()
        if total > 0:
            nuc_attr /= total
        return nuc_attr

    logger.info("Running IG for Head 1 (on-target, %d samples)...", len(samples_on))
    head1 = _run_ig(samples_on["grna_23mer"].tolist(), None, "ontarget")

    logger.info("Running IG for Head 2 (off-target, %d samples)...", len(samples_off))
    head2 = _run_ig(samples_off["grna"].tolist(), samples_off["dna_target"].tolist(), "offtarget")

    return {"head1_importance": head1, "head2_importance": head2}


# ─── Saliency visualization ────────────────────────────────────────────────────

def plot_saliency_comparison(ig_results: Dict, save_dir: Path) -> None:
    """Plot side-by-side bar charts of per-nucleotide IG importance.

    Seed region (positions 12-20, 0-indexed) highlighted in a different colour.
    Saves saliency_comparison.png in save_dir.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    head1 = ig_results["head1_importance"]
    head2 = ig_results["head2_importance"]
    n     = len(head1)
    positions = np.arange(1, n + 1)

    seed_start = 12  # config['interpretability']['seed_region_start']
    seed_end   = 20  # config['interpretability']['seed_region_end']

    def _bar_colors(n_pos: int) -> List:
        return [
            "#e05252" if seed_start <= i < seed_end else "#5b8dd9"
            for i in range(n_pos)
        ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, attr, title in zip(
        axes,
        [head1, head2],
        ["Head 1 — On-Target Efficiency", "Head 2 — Off-Target Activity"],
    ):
        colors = _bar_colors(n)
        ax.bar(positions, attr, color=colors, edgecolor="white", linewidth=0.4)
        ax.axvspan(seed_start + 1, seed_end + 1, alpha=0.08, color="#e05252",
                   label=f"Seed region ({seed_start+1}–{seed_end})")
        ax.set_xlabel("Nucleotide position (5'→3')", fontsize=11)
        ax.set_ylabel("Normalized IG attribution", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(positions)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("CRISPR-MTL: Integrated Gradients Saliency Comparison", fontsize=14)
    plt.tight_layout()

    out_path = save_dir / "saliency_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saliency plot saved → %s", out_path)
