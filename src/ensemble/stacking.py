"""
src/ensemble/stacking.py
Stacking Ensemble:
  Base models  : Logistic Regression + SVM + KNN + XGBoost
  Meta-learner : Logistic Regression (trained trên OOF predictions)

Kỹ thuật Out-of-Fold (OOF) để tránh data leakage:
  ┌──────────────────────────────────────────────────────────────┐
  │ 1. Với mỗi fold (cross_val_predict):                         │
  │      - Train base models trên K-1 folds                     │
  │      - Predict trên fold còn lại → OOF predictions          │
  │ 2. Stack OOF predictions thành meta-features [n×4]           │
  │ 3. Train Logistic Regression (meta-learner) trên meta-features│
  │ 4. Retrain tất cả base models trên TOÀN BỘ training data     │
  │ 5. Inference: base predictions → meta-learner → final output │
  └──────────────────────────────────────────────────────────────┘

Tại sao OOF?
  Nếu train base model và meta-learner trên CÙNG data, meta-learner
  sẽ học được "errors" của base models, dẫn đến overfitting.
  OOF đảm bảo meta-features được tạo từ unseen data.
"""

import time
import numpy as np
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from ..models.base_models import (
    build_logistic_regression,
    build_svm,
    build_knn,
    build_xgboost,
)


class StackingEnsemble:
    """
    Stacking Ensemble: LR + SVM + KNN + XGBoost → Logistic Regression (meta).

    Parameters (từ config['stacking']):
        random_state : int
        cv_folds     : int  - số folds OOF (mặc định 5)
    """

    def __init__(self, cfg: dict):
        self.cfg          = cfg
        self.random_state = cfg['stacking']['random_state']
        self.cv_folds     = cfg['stacking'].get('cv_folds', 5)

        # Populated after fit()
        self.lr_   = None   # base Logistic Regression
        self.svm_  = None
        self.knn_  = None
        self.xgb_  = None
        self.meta_ = None   # Logistic Regression meta-learner

        # Thời gian huấn luyện RIÊNG từng model (giây), populated sau fit()
        # Bao gồm: thời gian sinh OOF (cross_val_predict) + thời gian retrain
        # trên toàn bộ train set — tức tổng chi phí thực tế để có 1 model sẵn
        # sàng dùng, tương đương cách đo nếu train model đó độc lập.
        self.model_times_ = {}

    # ─── Build models từ config ──────────────────────────────────────────────

    def _build_models(self):
        lr_cfg  = self.cfg['models']['logistic_regression']
        svm_cfg = self.cfg['models']['svm']
        knn_cfg = self.cfg['models']['knn']
        xgb_cfg = self.cfg['models']['xgboost']

        self.lr_ = build_logistic_regression(
            C            = lr_cfg.get('C', 1.0),
            max_iter     = lr_cfg.get('max_iter', 1000),
            random_state = lr_cfg.get('random_state', self.random_state),
        )
        self.svm_ = build_svm(
            kernel       = svm_cfg.get('kernel', 'rbf'),
            C            = svm_cfg.get('C', 1.0),
            random_state = svm_cfg.get('random_state', self.random_state),
        )
        self.knn_ = build_knn(
            n_neighbors  = knn_cfg.get('n_neighbors', 5),
        )
        self.xgb_ = build_xgboost(
            n_estimators  = xgb_cfg.get('n_estimators', 100),
            max_depth     = xgb_cfg.get('max_depth', 4),
            learning_rate = xgb_cfg.get('learning_rate', 0.1),
            random_state  = xgb_cfg.get('random_state', self.random_state),
        )
        self.meta_ = LogisticRegression(
            C            = 1.0,
            max_iter     = 1000,
            solver       = 'lbfgs',
            random_state = self.random_state,
            # n_jobs không dùng cho LR (deprecated từ sklearn 1.8)
        )

    # ─── Fit ─────────────────────────────────────────────────────────────────

    def fit(self, X_train, y_train):
        """
        Train stacking ensemble bằng kỹ thuật Out-of-Fold.
        X_train, y_train đã qua SMOTE và RFECV feature selection.
        """
        self._build_models()

        skf = StratifiedKFold(
            n_splits     = self.cv_folds,
            shuffle      = True,
            random_state = self.random_state,
        )

        print(f"[Stacking] Generating OOF meta-features (cv_folds={self.cv_folds}) ...")

        t0 = time.time()
        oof_lr = cross_val_predict(
            self.lr_, X_train, y_train,
            cv=skf, method='predict_proba', n_jobs=1
        )[:, 1]
        t_lr_oof = time.time() - t0
        print("[Stacking] ✓ Logistic Regression OOF done")

        t0 = time.time()
        oof_svm = cross_val_predict(
            self.svm_, X_train, y_train,
            cv=skf, method='predict_proba', n_jobs=1
        )[:, 1]
        t_svm_oof = time.time() - t0
        print("[Stacking] ✓ SVM OOF done")

        t0 = time.time()
        oof_knn = cross_val_predict(
            self.knn_, X_train, y_train,
            cv=skf, method='predict_proba', n_jobs=1
        )[:, 1]
        t_knn_oof = time.time() - t0
        print("[Stacking] ✓ KNN OOF done")

        t0 = time.time()
        oof_xgb = cross_val_predict(
            self.xgb_, X_train, y_train,
            cv=skf, method='predict_proba', n_jobs=1
        )[:, 1]
        t_xgb_oof = time.time() - t0
        print("[Stacking] ✓ XGBoost OOF done")

        # Stack thành meta-feature matrix [n_samples × 4]
        meta_X_train = np.column_stack([oof_lr, oof_svm, oof_knn, oof_xgb])

        # Train Logistic Regression meta-learner trên OOF predictions
        print("[Stacking] Training Logistic Regression meta-learner ...")
        t0 = time.time()
        self.meta_.fit(meta_X_train, y_train)
        t_meta = time.time() - t0

        # Retrain base models trên toàn bộ training data
        print("[Stacking] Retraining base models on full train set ...")
        t0 = time.time()
        self.lr_.fit(X_train, y_train)
        t_lr_refit = time.time() - t0

        t0 = time.time()
        self.svm_.fit(X_train, y_train)
        t_svm_refit = time.time() - t0

        t0 = time.time()
        self.knn_.fit(X_train, y_train)
        t_knn_refit = time.time() - t0

        t0 = time.time()
        self.xgb_.fit(X_train, y_train)
        t_xgb_refit = time.time() - t0

        # Thời gian "thực" mỗi model = OOF (cv_folds lần fit) + refit cuối
        # (đây là toàn bộ chi phí tính toán thực sự dùng cho model đó trong
        # pipeline Stacking, không phải con số dùng chung cho cả cụm)
        self.model_times_ = {
            'Logistic Regression': t_lr_oof  + t_lr_refit,
            'SVM':                 t_svm_oof + t_svm_refit,
            'KNN':                 t_knn_oof + t_knn_refit,
            'XGBoost':              t_xgb_oof + t_xgb_refit,
            'Stacking Ensemble':   (t_lr_oof + t_svm_oof + t_knn_oof + t_xgb_oof
                                     + t_meta
                                     + t_lr_refit + t_svm_refit + t_knn_refit + t_xgb_refit),
        }

        print("[Stacking] ✓ Fit complete.")
        return self

    # ─── Predict ─────────────────────────────────────────────────────────────

    def _make_meta_features(self, X) -> np.ndarray:
        """Tạo meta-features từ predictions của 4 base models."""
        lr_pr  = self.lr_.predict_proba(X)[:, 1]
        svm_pr = self.svm_.predict_proba(X)[:, 1]
        knn_pr = self.knn_.predict_proba(X)[:, 1]
        xgb_pr = self.xgb_.predict_proba(X)[:, 1]
        return np.column_stack([lr_pr, svm_pr, knn_pr, xgb_pr])

    def predict(self, X) -> np.ndarray:
        meta_X = self._make_meta_features(X)
        return self.meta_.predict(meta_X)

    def predict_proba(self, X) -> np.ndarray:
        meta_X = self._make_meta_features(X)
        return self.meta_.predict_proba(meta_X)

    # ─── Expose base model predictions (for comparison table) ────────────────

    def predict_all_models(self, X) -> dict:
        """
        Trả về predictions của tất cả base models + stacking ensemble.
        Dùng để tạo bảng so sánh full.
        """
        return {
            'Logistic Regression': {
                'pred': self.lr_.predict(X),
                'prob': self.lr_.predict_proba(X)[:, 1],
            },
            'SVM': {
                'pred': self.svm_.predict(X),
                'prob': self.svm_.predict_proba(X)[:, 1],
            },
            'KNN': {
                'pred': self.knn_.predict(X),
                'prob': self.knn_.predict_proba(X)[:, 1],
            },
            'XGBoost': {
                'pred': self.xgb_.predict(X),
                'prob': self.xgb_.predict_proba(X)[:, 1],
            },
            'Stacking Ensemble': {
                'pred': self.predict(X),
                'prob': self.predict_proba(X)[:, 1],
            },
        }