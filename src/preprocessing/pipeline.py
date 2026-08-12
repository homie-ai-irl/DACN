"""
src/preprocessing/pipeline.py
Preprocessing pipeline:
  - Preprocessor  : StandardScaler (chỉ fit trên train, transform cả train/test)
  - SmoteBalancer : SMOTE oversampling (chỉ áp dụng trên train sau khi split)

Lưu ý quan trọng về thứ tự:
  raw data → StandardScaler → Train/Test Split → SMOTE (chỉ train) → RFECV → Training
  Test set KHÔNG bao giờ qua SMOTE để tránh data leakage.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


class Preprocessor:
    """
    Single-stage StandardScaler normalization.
    fit_transform() chỉ gọi trên X_train,
    transform()     gọi trên cả X_train (sau fit) và X_test.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)


class SmoteBalancer:
    """
    SMOTE (Synthetic Minority Over-sampling Technique).
    - Chỉ áp dụng trên tập TRAIN sau khi đã split.
    - Tập test giữ nguyên distribution gốc → đánh giá thực tế hơn.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.smote = SMOTE(random_state=random_state)

    def fit_resample(self, X_train: np.ndarray, y_train: np.ndarray):
        counts_before = dict(zip(*np.unique(y_train, return_counts=True)))
        print(f"[SMOTE] Before: {counts_before}")
        X_res, y_res = self.smote.fit_resample(X_train, y_train)
        counts_after = dict(zip(*np.unique(y_res, return_counts=True)))
        print(f"[SMOTE] After : {counts_after}")
        ratio = counts_after.get(1, 0) / counts_after.get(0, 1)
        print(f"[SMOTE] Fraud/Normal ratio after: {ratio:.3f}")
        return X_res, y_res
