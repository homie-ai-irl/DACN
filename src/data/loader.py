"""
src/data/loader.py
Load và chia train/test dataset creditcard.csv
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str):
    """Load creditcard.csv, trả về X, y, feature_names."""
    print(f"[Data] Loading: {path}")
    df = pd.read_csv(path)
    print(f"[Data] Shape: {df.shape}")
    counts = df['Class'].value_counts().to_dict()
    print(f"[Data] Class distribution: {counts}")
    fraud_pct = df['Class'].sum() / len(df) * 100
    print(f"[Data] Fraud rate: {fraud_pct:.4f}%")

    feature_names = [c for c in df.columns if c != 'Class']
    X = df[feature_names].values
    y = df['Class'].values
    return X, y, feature_names


def subsample_majority(X, y, majority_n: int, random_state: int = 42):
    """
    Giảm số lượng class 0 xuống majority_n (giữ nguyên toàn bộ class 1).
    Cần thiết khi SMOTE trên tập quá lớn (284k) quá chậm.
    """
    if majority_n is None:
        return X, y

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    if len(idx0) <= majority_n:
        print(f"[Data] Majority class already <= {majority_n}, no subsampling needed.")
        return X, y

    rng = np.random.RandomState(random_state)
    chosen0 = rng.choice(idx0, size=majority_n, replace=False)
    idx_all = np.concatenate([chosen0, idx1])
    rng.shuffle(idx_all)

    print(f"[Data] Subsampled majority: {majority_n} | Fraud kept: {len(idx1)}")
    return X[idx_all], y[idx_all]


def split_data(X, y, test_size: float = 0.20, random_state: int = 42):
    """Stratified train/test split — TRƯỚC SMOTE."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Split] Train={len(y_tr)} | Test={len(y_te)}")
    print(f"[Split] Train fraud: {y_tr.sum()} ({y_tr.mean()*100:.2f}%)")
    print(f"[Split] Test  fraud: {y_te.sum()} ({y_te.mean()*100:.2f}%)")
    return X_tr, X_te, y_tr, y_te
