# Data

This folder contains the three datasets used in the CRISPR-MTL research project. All datasets are sourced from public repositories. **Do not commit raw data files to GitHub** — add them manually following the instructions below.

---

## Folder Structure

```
data/
├── on-target/
│   └── Supplementary File1.csv
└── off-target/
    ├── eg_cls_off_target.epiotrt
    └── listgarten_elevation_hmg.pkl
```

---

## Dataset 1 — On-Target: Doench 2016

**File:** `on-target/Supplementary File1.csv`
**Size:** ~504 KB | **Samples:** 5,310 gRNA sequences

The most widely used on-target benchmark dataset in the CRISPR literature. Contains guide sequences from 17 genes tested in human cells via high-throughput plasmid library experiments.

| Column | Description |
|---|---|
| `30mer` | 30-nucleotide DNA sequence (gRNA + flanking context) |
| `predictions` | Efficiency label (normalized indel frequency, range 0–1) |

**Preprocessing note:** The 23-mer input for DNABERT is extracted as `30mer[4:27]` — 20 nt gRNA + 3 nt PAM sequence.

**How to obtain:**
```bash
git clone https://github.com/khaled-buet/CRISPRpred.git
cp CRISPRpred/CRISPRpred/SupplementaryFiles/"Supplementary File1.csv" data/on-target/
```

**Reference:** Doench et al. (2016). *Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9.* Nature Biotechnology, 34(2), 184–191.

---

## Dataset 2 — Off-Target: DeepCRISPR Benchmark

**File:** `off-target/eg_cls_off_target.epiotrt`
**Format:** Serialized object (pickle-compatible)

A compiled off-target benchmark dataset used in the DeepCRISPR paper, aggregated from multiple GUIDE-seq and CIRCLE-seq experiments. Each sample is an sgRNA-DNA pair with a binary label (1 = confirmed off-target cleavage, 0 = no cleavage).

**How to obtain:**
```bash
git clone https://github.com/dagrate/public_data_crisprCas9.git
cp public_data_crisprCas9/data/deepcrispr/eg_cls_off_target.epiotrt data/off-target/
```

**Reference:** Chuai et al. (2018). *DeepCRISPR: optimized CRISPR guide RNA design by deep learning.* Genome Biology, 19(1), 80.

---

## Dataset 3 — Off-Target: Listgarten Elevation GUIDE-seq

**File:** `off-target/listgarten_elevation_hmg.pkl`
**Format:** sklearn Bunch object (pickle)

A GUIDE-seq dataset from genome-wide Cas9 cleavage experiments in human cells, published alongside the Elevation model. Used as a secondary off-target source merged with Dataset 2 to enrich the training distribution.

**How to obtain:**
```bash
# From the same repository as Dataset 2
cp public_data_crisprCas9/data/listgarten_elevation_hmg/listgarten_elevation_hmg.pkl data/off-target/
```

**Reference:** Listgarten et al. (2018). *Prediction of off-target activities for the end-to-end design of CRISPR guide RNAs.* Nature Biomedical Engineering, 2(1), 38–47.

---

## Loading the Datasets

Run the preprocessing script to load, clean, and prepare all datasets:

```bash
python src/dataset.py
```

This will generate:
```
data/processed/
├── ontarget_clean.csv       # cleaned on-target dataset
├── offtarget_merged.csv     # merged and deduplicated off-target dataset
└── cv_splits.pkl            # 5-fold cross-validation indices
```

Or load manually for quick exploration:

```python
import pandas as pd
import pickle

# Dataset 1: On-Target
df_on = pd.read_csv('data/on-target/Supplementary File1.csv')
df_on['grna_23mer'] = df_on['30mer'].str[4:27]
df_on['label']      = df_on['predictions'].astype(float)

# Dataset 2: Off-Target (DeepCRISPR)
with open('data/off-target/eg_cls_off_target.epiotrt', 'rb') as f:
    data_deepcrispr = pickle.load(f)

# Dataset 3: Off-Target (Listgarten)
with open('data/off-target/listgarten_elevation_hmg.pkl', 'rb') as f:
    data_listgarten = pickle.load(f)
```

---

## Notes

- Raw data files are excluded from version control via `.gitignore`.
- Datasets 2 and 3 are merged and deduplicated (by sgRNA-DNA pair) before training.
- Class imbalance in the off-target data is handled with `w_pos = n_negative / n_positive` in the BCE loss.
- All three datasets are used consistently across all experimental runs with the same random seed (`seed=42`) and 5-fold stratified cross-validation splits.