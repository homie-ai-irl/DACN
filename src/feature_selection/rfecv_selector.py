"""
src/feature_selection/rfecv_selector.py
Hai bộ chọn đặc trưng:

  RFEFixedSelector  – RFE với n_features_to_select CỐ ĐỊNH (15 hoặc 20)
                      Nhanh hơn RFECV vì không cần cross-validation.
                      Phù hợp khi muốn kiểm soát chính xác số lượng features.

  RFECVSelector     – RFECV tự động tìm n_features tối ưu qua cross-validation.
                      Chậm hơn nhưng cho kết quả tốt hơn về mặt lý thuyết.

Estimator hỗ trợ: 'xgboost' | 'random_forest'

Quy trình RFE:
  1. Fit estimator → lấy feature_importances_
  2. Loại bỏ feature ít quan trọng nhất
  3. Lặp lại đến khi còn n_features_to_select features
"""

import numpy as np
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Helper – build estimator dùng chung cho cả 2 class
# ─────────────────────────────────────────────────────────────────────────────

def _build_estimator(name: str, random_state: int = 42):
    """Tạo estimator theo tên."""
    if name == 'xgboost':
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost chưa được cài. Chạy: pip install xgboost")
        return XGBClassifier(
            n_estimators  = 100,
            max_depth     = 4,
            learning_rate = 0.1,
            subsample     = 0.8,
            eval_metric   = 'logloss',
            random_state  = random_state,
            n_jobs        = 1,
        )
    elif name == 'random_forest':
        return RandomForestClassifier(
            n_estimators = 100,
            max_depth    = None,
            random_state = random_state,
            n_jobs       = 1,
        )
    else:
        raise ValueError(
            f"Unknown estimator '{name}'. "
            "Dùng 'xgboost' hoặc 'random_forest'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# RFEFixedSelector – n_features CỐ ĐỊNH (15 hoặc 20)
# ─────────────────────────────────────────────────────────────────────────────

class RFEFixedSelector:
    """
    RFE với số features cố định (không cần cross-validation).

    Parameters
    ----------
    n_features   : int   – Số features muốn giữ lại (ví dụ 15 hoặc 20)
    estimator    : str   – 'xgboost' hoặc 'random_forest'
    step         : int|float – Số features loại mỗi bước (1 = loại từng cái)
    random_state : int
    """

    def __init__(
        self,
        n_features   : int   = 20,
        estimator    : str   = 'xgboost',
        step         : int   = 1,
        random_state : int   = 42,
    ):
        self.n_features    = n_features
        self.estimator_name = estimator
        self.step          = step
        self.random_state  = random_state

        # Populated after fit()
        self.rfe_              = None
        self.selected_indices_ = None
        self.selected_names_   = None
        self.feature_names_    = None

    def fit(self, X_train, y_train, feature_names=None):
        n_feat = X_train.shape[1]
        # Đảm bảo n_features không vượt quá số features thực tế
        n_select = min(self.n_features, n_feat)

        self.feature_names_ = (
            feature_names if feature_names is not None
            else [f'feature_{i}' for i in range(n_feat)]
        )

        estimator = _build_estimator(self.estimator_name, self.random_state)

        self.rfe_ = RFE(
            estimator              = estimator,
            n_features_to_select   = n_select,
            step                   = self.step,
            verbose                = 0,
        )

        print(f"[RFE-Fixed] Estimator   : {self.estimator_name.upper()}")
        print(f"[RFE-Fixed] n_features  : {n_select} / {n_feat}")
        print(f"[RFE-Fixed] Fitting on {X_train.shape[0]} samples × {n_feat} features ...")

        self.rfe_.fit(X_train, y_train)

        self.selected_indices_ = np.where(self.rfe_.support_)[0]
        self.selected_names_   = [self.feature_names_[i]
                                   for i in self.selected_indices_]

        print(f"[RFE-Fixed] Selected    : {self.selected_names_}")
        return self

    def transform(self, X) -> np.ndarray:
        return X[:, self.selected_indices_]

    def fit_transform(self, X_train, y_train, feature_names=None) -> np.ndarray:
        self.fit(X_train, y_train, feature_names)
        return self.transform(X_train)

    def get_selected_names(self):
        return self.selected_names_

    def get_feature_ranking(self):
        """List (feature_name, ranking). ranking=1 → được chọn."""
        return list(zip(self.feature_names_, self.rfe_.ranking_))

    def get_feature_importances(self):
        """
        Dict {feature_name: importance} từ estimator bên trong RFE.
        Chỉ có với tree-based estimators (RF, XGBoost).
        """
        est = self.rfe_.estimator_
        if not hasattr(est, 'feature_importances_'):
            return {}
        imp = est.feature_importances_
        # imp có độ dài = n_features đã chọn (sklearn fit estimator trên subset)
        return dict(zip(self.selected_names_, imp))

    # Compatibility với RFECVSelector để dùng chung trong plots
    @property
    def rfecv_(self):
        """Alias để tương thích với code dùng RFECVSelector."""
        return self.rfe_

    @property
    def estimator_name(self):
        return self._estimator_name

    @estimator_name.setter
    def estimator_name(self, v):
        self._estimator_name = v

    def get_cv_scores(self):
        return None   # RFE không có CV scores

    def get_std_scores(self):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RFECVSelector – tự động tìm n_features tối ưu qua cross-validation
# ─────────────────────────────────────────────────────────────────────────────

class RFECVSelector:
    """
    RFECV feature selector (tự động chọn n_features tối ưu).

    Parameters
    ----------
    estimator    : str  – 'xgboost' hoặc 'random_forest'
    cv           : int  – Số folds cross-validation
    scoring      : str  – Metric tối ưu ('f1', 'roc_auc', 'average_precision')
    min_features : int  – Số features tối thiểu
    random_state : int
    """

    def __init__(
        self,
        estimator    : str = 'xgboost',
        cv           : int = 5,
        scoring      : str = 'f1',
        min_features : int = 5,
        random_state : int = 42,
    ):
        self.estimator_name = estimator
        self.cv             = cv
        self.scoring        = scoring
        self.min_features   = min_features
        self.random_state   = random_state

        self.rfecv_            = None
        self.selected_indices_ = None
        self.selected_names_   = None
        self.feature_names_    = None

    def fit(self, X_train, y_train, feature_names=None):
        n_feat = X_train.shape[1]
        self.feature_names_ = (
            feature_names if feature_names is not None
            else [f'feature_{i}' for i in range(n_feat)]
        )

        estimator   = _build_estimator(self.estimator_name, self.random_state)
        cv_strategy = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        self.rfecv_ = RFECV(
            estimator              = estimator,
            step                   = 1,
            cv                     = cv_strategy,
            scoring                = self.scoring,
            min_features_to_select = self.min_features,
            n_jobs                 = 1,
        )

        print(f"[RFECV] Estimator   : {self.estimator_name.upper()}")
        print(f"[RFECV] CV folds    : {self.cv}  |  Scoring: {self.scoring}")
        print(f"[RFECV] Min features: {self.min_features}")
        print(f"[RFECV] Fitting on {X_train.shape[0]} × {n_feat} ...")

        self.rfecv_.fit(X_train, y_train)

        self.selected_indices_ = np.where(self.rfecv_.support_)[0]
        self.selected_names_   = [self.feature_names_[i]
                                   for i in self.selected_indices_]

        print(f"[RFECV] Optimal n_features : {self.rfecv_.n_features_} / {n_feat}")
        print(f"[RFECV] Selected features  : {self.selected_names_}")
        return self

    def transform(self, X) -> np.ndarray:
        return X[:, self.selected_indices_]

    def fit_transform(self, X_train, y_train, feature_names=None) -> np.ndarray:
        self.fit(X_train, y_train, feature_names)
        return self.transform(X_train)

    def get_selected_names(self):
        return self.selected_names_

    def get_feature_ranking(self):
        return list(zip(self.feature_names_, self.rfecv_.ranking_))

    def get_feature_importances(self):
        est = self.rfecv_.estimator_
        if not hasattr(est, 'feature_importances_'):
            return {}
        imp = est.feature_importances_
        return dict(zip(self.feature_names_, imp))

    def get_cv_scores(self):
        return self.rfecv_.cv_results_['mean_test_score']

    def get_std_scores(self):
        return self.rfecv_.cv_results_.get(
            'std_test_score',
            np.zeros_like(self.get_cv_scores())
        )
