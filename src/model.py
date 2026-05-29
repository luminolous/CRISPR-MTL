"""Model classes for CRISPR-MTL.

Four architectures:
  BiLSTMBaseline      — Experiment A1 (on-target, scratch)
  CNNBiLSTMBaseline   — Experiment A2 (off-target, scratch)
  DNABERTSingleTask   — Experiments B1, B2
  CRISPRMultiTask     — MTL-Full, ABL1, ABL2, ABL3
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)


# ─── Shared building blocks ────────────────────────────────────────────────────

def _make_shared_projection(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Shared projection layer: Dropout → Linear → GELU → LayerNorm."""
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.LayerNorm(out_dim),
    )


def _make_task_head(in_dim: int, dropout: float) -> nn.Sequential:
    """Task head: Linear(in→64) → ReLU → Dropout → Linear(64→1) → Sigmoid."""
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


# ─── Freeze utilities ─────────────────────────────────────────────────────────

def _apply_freeze_strategy(bert: nn.Module, strategy: str) -> None:
    """Apply a layer freeze strategy to a HuggingFace BertModel.

    'freeze_all'  : freeze all DNABERT parameters
    'freeze_8'    : freeze embeddings + layers 0-7; unfreeze layers 8-11
    'unfreeze_all': unfreeze all DNABERT parameters
    """
    # Freeze everything first
    for param in bert.parameters():
        param.requires_grad = False

    if strategy == "freeze_all":
        pass  # done

    elif strategy == "freeze_8":
        # Unfreeze transformer layers 8-11 (CLAUDE.md: layers 9-12, 1-indexed)
        n_layers = len(bert.encoder.layer)
        for i in range(8, n_layers):
            for param in bert.encoder.layer[i].parameters():
                param.requires_grad = True
        # Pooler (if present) also unfrozen for fine-tuning
        if hasattr(bert, "pooler") and bert.pooler is not None:
            for param in bert.pooler.parameters():
                param.requires_grad = True

    elif strategy == "unfreeze_all":
        for param in bert.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown freeze strategy: {strategy!r}. "
                         f"Choose from 'freeze_all', 'freeze_8', 'unfreeze_all'.")


# ─── BiLSTMBaseline (Experiment A1) ───────────────────────────────────────────

class BiLSTMBaseline(nn.Module):
    """BiLSTM from scratch for on-target efficiency (regression). Experiment A1.

    Input:  (batch, seq_len=23, 4)  — one-hot encoded gRNA
    Output: (batch, 1)              — predicted efficiency in [0, 1]
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        lstm_hidden = config["baseline"]["lstm_hidden"]  # 128
        lstm_layers = config["baseline"]["lstm_layers"]  # 2

        self.input_proj = nn.Linear(4, 32)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 23, 4) → (batch, 1)."""
        x = self.input_proj(x)          # (batch, 23, 32)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers*2, batch, lstm_hidden)
        # Last layer: forward = h_n[-2], backward = h_n[-1]
        out = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, lstm_hidden*2)
        return self.head(out)           # (batch, 1)


# ─── CNNBiLSTMBaseline (Experiment A2) ────────────────────────────────────────

class CNNBiLSTMBaseline(nn.Module):
    """CNN-BiLSTM from scratch for off-target activity (classification). Experiment A2.

    Input:  (batch, seq_len=23, 7)  — 7-channel mismatch matrix
    Output: (batch, 1)              — predicted activity in [0, 1]

    7-channel encoding (see dataset.encode_mismatch):
        ch 0-3 : gRNA one-hot (A, C, G, T)
        ch 4   : exact match at position
        ch 5   : transition mismatch (A↔G or C↔T)
        ch 6   : transversion mismatch (purine↔pyrimidine)
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        cnn_filters = config["baseline"]["cnn_filters"]             # 64
        cnn_kernel  = config["baseline"]["cnn_kernel"]              # 3
        lstm_hidden = config["baseline"]["lstm_hidden"]             # 128
        drop_off    = config["model"]["head_dropout_offtarget"]     # 0.3

        self.conv = nn.Sequential(
            nn.Conv1d(7, cnn_filters, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(drop_off),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 23, 7) → (batch, 1)."""
        x = x.permute(0, 2, 1)           # (batch, 7, 23)
        x = self.conv(x)                 # (batch, cnn_filters, ~12)
        x = x.permute(0, 2, 1)           # (batch, ~12, cnn_filters)
        _, (h_n, _) = self.lstm(x)
        out = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, lstm_hidden*2)
        return self.head(out)            # (batch, 1)


# ─── DNABERTSingleTask (Experiments B1, B2) ───────────────────────────────────

class DNABERTSingleTask(nn.Module):
    """DNABERT encoder with a single task head.

    Experiment B1: task='ontarget'  (regression)
    Experiment B2: task='offtarget' (classification)

    Freeze strategy on init: 'freeze_8' (layers 1-8 frozen, 9-12 trainable).
    Call model.freeze_strategy('freeze_all') for Phase-1 warmup.
    """

    def __init__(
        self,
        config: dict,
        task: str,
        _bert: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert task in ("ontarget", "offtarget"), f"Unknown task: {task!r}"
        self.task = task

        hidden    = config["model"]["dnabert_hidden"]        # 768
        proj_dim  = config["model"]["projection_dim"]        # 256
        proj_drop = config["model"]["projection_dropout"]    # 0.1

        # DNABERT backbone (or injected mock for testing)
        if _bert is not None:
            self.bert = _bert
        else:
            logger.info("Loading DNABERT from %s", config["model"]["dnabert_name"])
            self.bert = AutoModel.from_pretrained(config["model"]["dnabert_name"])

        # Shared projection
        self.projection = _make_shared_projection(hidden, proj_dim, proj_drop)

        # Task head
        head_drop = (
            config["model"]["head_dropout_ontarget"]
            if task == "ontarget"
            else config["model"]["head_dropout_offtarget"]
        )
        self.head = _make_task_head(proj_dim, head_drop)

        # Default: freeze layers 1-8, unfreeze 9-12
        self.freeze_strategy("freeze_8")

    def freeze_strategy(self, strategy: str) -> None:
        """Apply layer freeze strategy. See _apply_freeze_strategy for options."""
        _apply_freeze_strategy(self.bert, strategy)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """input_ids, attention_mask: (batch, seq_len) → (batch, 1)."""
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = bert_out.last_hidden_state[:, 0, :]   # (batch, 768)
        x   = self.projection(cls)                   # (batch, 256)
        return self.head(x)                          # (batch, 1)


# ─── CRISPRMultiTask (MTL-Full, ABL1, ABL2, ABL3) ────────────────────────────

class CRISPRMultiTask(nn.Module):
    """Shared DNABERT encoder with dual task heads. Main model + all ablations.

    MTL-Full : alternating batch training, phase-based unfreezing
    ABL1     : freeze_strategy='freeze_all' throughout
    ABL2     : freeze_strategy='unfreeze_all' in Phase 2
    ABL3     : combined loss (handled in train.py, not here)
    """

    def __init__(
        self,
        config: dict,
        _bert: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        hidden    = config["model"]["dnabert_hidden"]        # 768
        proj_dim  = config["model"]["projection_dim"]        # 256
        proj_drop = config["model"]["projection_dropout"]    # 0.1

        # DNABERT backbone
        if _bert is not None:
            self.bert = _bert
        else:
            logger.info("Loading DNABERT from %s", config["model"]["dnabert_name"])
            self.bert = AutoModel.from_pretrained(config["model"]["dnabert_name"])

        # Shared projection (both tasks feed through this)
        self.projection = _make_shared_projection(hidden, proj_dim, proj_drop)

        # Task heads
        self.head_ontarget  = _make_task_head(proj_dim, config["model"]["head_dropout_ontarget"])   # 0.2
        self.head_offtarget = _make_task_head(proj_dim, config["model"]["head_dropout_offtarget"])  # 0.3

        # Default: freeze layers 1-8, unfreeze 9-12
        self.freeze_strategy("freeze_8")

    def freeze_strategy(self, strategy: str) -> None:
        """Apply layer freeze strategy. See _apply_freeze_strategy for options."""
        _apply_freeze_strategy(self.bert, strategy)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """input_ids, attention_mask: (batch, seq_len), task: 'ontarget'|'offtarget' → (batch, 1)."""
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = bert_out.last_hidden_state[:, 0, :]   # (batch, 768)
        x   = self.projection(cls)                   # (batch, 256)

        if task == "ontarget":
            return self.head_ontarget(x)             # (batch, 1)
        elif task == "offtarget":
            return self.head_offtarget(x)            # (batch, 1)
        else:
            raise ValueError(f"Unknown task: {task!r}. Use 'ontarget' or 'offtarget'.")
