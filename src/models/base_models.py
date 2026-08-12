"""
src/models/base_models.py
Factory functions cho các base ML models:
  - Logistic Regression
  - SVM (Support Vector Machine)
  - KNN (K-Nearest Neighbors)
  - XGBoost

Tất cả model đều:
  - n_jobs=1  → tránh lỗi multiprocessing trên Windows
  - probability=True (SVM) → hỗ trợ predict_proba cho ROC/PR curves
"""

from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


def build_logistic_regression(C=1.0, max_iter=1000, random_state=42):
    """
    Logistic Regression.
    solver='lbfgs' tốt cho binary classification với regularization L2.
    """
    return LogisticRegression(
        C            = C,
        max_iter     = max_iter,
        solver       = 'lbfgs',
        random_state = random_state,
    )


def build_svm(kernel='rbf', C=1.0, random_state=42):
    """
    SVM với probability=True (Platt scaling) để hỗ trợ predict_proba.
    kernel='rbf' chuẩn xác hơn 'linear', nhưng chậm hơn trên large data.
    """
    return SVC(
        kernel       = kernel,
        C            = C,
        gamma        = 'scale',
        probability  = True,
        random_state = random_state,
    )


def build_knn(n_neighbors=5):
    """K-Nearest Neighbors."""
    return KNeighborsClassifier(
        n_neighbors = n_neighbors,
        metric      = 'minkowski',
        n_jobs      = 1,
    )


def build_xgboost(n_estimators=100, max_depth=4, learning_rate=0.1,
                  scale_pos_weight=1, random_state=42):
    """
    XGBoost.
    scale_pos_weight: tự động set bằng (n_negative / n_positive) nếu muốn
    xử lý class imbalance mà không cần SMOTE (nhưng ta đã có SMOTE rồi).
    """
    if not _XGB_AVAILABLE:
        raise ImportError("xgboost chưa được cài. Chạy: pip install xgboost")
    return XGBClassifier(
        n_estimators     = n_estimators,
        max_depth        = max_depth,
        learning_rate    = learning_rate,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale_pos_weight,
        eval_metric      = 'logloss',
        random_state     = random_state,
        n_jobs           = 1,
    )
