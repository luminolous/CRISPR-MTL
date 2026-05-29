"""Temporary script to explore raw dataset structure before writing preprocessing code."""

import copy
import pickle as pkl
import pandas as pd
import pandas.compat.pickle_compat as pc
from pandas.core.internals import BlockManager
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
ON_TARGET_PATH = ROOT / "data" / "on-target" / "Supplementary File1.csv"
DC_PATH = ROOT / "data" / "off-target" / "eg_cls_off_target.epiotrt"
LG_PATH = ROOT / "data" / "off-target" / "listgarten_elevation_hmg.pkl"


def _patch_pickle_compat() -> None:
    """Patch pandas Unpickler to handle BlockManager from pandas < 2.0 pickles."""
    def patched_load_reduce(self) -> None:
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

    pc.Unpickler.dispatch = copy.copy(pc.Unpickler.dispatch)
    pc.Unpickler.dispatch[pkl.REDUCE[0]] = patched_load_reduce


_patch_pickle_compat()


def explore_csv(path: Path) -> None:
    print("=" * 60)
    print(f"ON-TARGET: {path.name}")
    print("=" * 60)
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")

    if "predictions" in df.columns:
        p = df["predictions"]
        print(f"\npredictions range: min={p.min():.4f}, max={p.max():.4f}, mean={p.mean():.4f}, std={p.std():.4f}")
    else:
        print("\nWARN: 'predictions' column not found")

    if "30mer" in df.columns:
        lengths = df["30mer"].str.len()
        print(f"\n30mer length distribution:\n{lengths.value_counts().sort_index().to_string()}")
    else:
        print("\nWARN: '30mer' column not found")


def explore_deepcrispr(path: Path) -> None:
    print("\n" + "=" * 60)
    print(f"OFF-TARGET (DeepCRISPR): {path.name}")
    print("=" * 60)
    # Tab-separated text file, no header
    df = pd.read_csv(path, sep="\t", header=None)
    print(f"Shape: {df.shape}")
    print(f"Columns (index): {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head(5).to_string()}")
    print(f"\nCol 0 (guide IDs) distribution:\n{df[0].value_counts().to_string()}")
    for col in df.columns:
        if df[col].dtype == object:
            lens = df[col].str.len().value_counts()
            print(f"Col {col} string lengths: {lens.to_dict()}")
    label_col = df.columns[-1]
    print(f"\nLabel (col {label_col}) distribution:\n{df[label_col].value_counts().to_string()}")
    print(f"\nKey columns: 0=guide_id, 1=gRNA(23mer), 6=DNA_target(23mer), {label_col}=label")


def explore_listgarten(path: Path) -> None:
    print("\n" + "=" * 60)
    print(f"OFF-TARGET (Listgarten): {path.name}")
    print("=" * 60)
    df = pc.load(open(path, "rb"), encoding="latin-1")
    print(f"Type: {type(df)}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head(5).to_string()}")
    for col in df.columns:
        if df[col].dtype == object:
            lens = df[col].dropna().str.len().value_counts()
            print(f"\n{col}: object, length dist={lens.to_dict()}, sample={repr(df[col].iloc[0])}")
        else:
            non_null = df[col].dropna()
            if len(non_null):
                print(f"\n{col}: {df[col].dtype}, min={non_null.min()}, max={non_null.max()}, nunique={non_null.nunique()}, null={df[col].isna().sum()}")
            else:
                print(f"\n{col}: {df[col].dtype}, ALL NULL")
    print(f"\nLabel (wasValidated) distribution:\n{df['wasValidated'].value_counts().to_string()}")
    print(f"\nKey columns: 30mer=gRNA(23mer), 30mer_mut=DNA_target(23mer), wasValidated=label")


if __name__ == "__main__":
    explore_csv(ON_TARGET_PATH)
    explore_deepcrispr(DC_PATH)
    explore_listgarten(LG_PATH)
