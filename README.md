# Credit Card Fraud Detection

Hệ thống phát hiện gian lận thẻ tín dụng dùng **Hybrid Feature Selection (RFE)** kết hợp **Stacking Ensemble Learning (Out-of-Fold)**. Project phục vụ đồ án/luận văn tốt nghiệp (DACN), tập trung chứng minh hai luận điểm:

1. **Feature Selection (RFE)** giúp giảm số chiều dữ liệu (30 → N features) trong khi vẫn giữ (hoặc cải thiện) hiệu năng phát hiện gian lận, đồng thời giảm thời gian huấn luyện — so sánh trực tiếp Before/After trên cùng pipeline.
2. **Stacking Ensemble** (Logistic Regression + SVM + KNN + XGBoost → meta-learner Logistic Regression, dùng kỹ thuật Out-of-Fold) cho kết quả tốt hơn từng mô hình đơn lẻ trên dữ liệu mất cân bằng nghiêm trọng (Fraud ≈ 0.17%).

---

## Mục lục

- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Pipeline tổng quan](#pipeline-tổng-quan)
- [Cấu trúc project](#cấu-trúc-project)
- [Cài đặt & Chạy](#cài-đặt--chạy)
- [Cấu hình](#cấu-hình-configsconfigyaml)
- [Feature Selection: RFE-Fixed](#feature-selection-rfe-fixed)
- [Outputs](#outputs-thư-mục-outputs)
- [Lý do chọn kỹ thuật](#lý-do-chọn-kỹ-thuật)
- [Dataset](#dataset)

---

## Yêu cầu hệ thống

| Thành phần       | Phiên bản tối thiểu |
|------------------|----------------------|
| Python           | 3.10+                |
| numpy            | ≥ 1.26.0             |
| pandas           | ≥ 2.1.0              |
| scikit-learn     | ≥ 1.4.0              |
| xgboost          | ≥ 2.0.0              |
| imbalanced-learn | ≥ 0.12.0             |
| matplotlib       | ≥ 3.8.0              |
| seaborn          | ≥ 0.13.0             |
| pyyaml           | ≥ 6.0                |
| Pillow           | ≥ 10.0.0             |

---

## Pipeline tổng quan

Đúng theo thứ tự thực thi trong `train.py`:

```
Credit Card Fraud Detection Dataset (creditcard.csv)
        │
        ▼
  [STEP 1] Load Data + Subsample majority class (tùy chọn, mặc định 50.000)
        │
        ▼
  [STEP 2] Train / Test Split (80% / 20%, stratified) → StandardScaler
           Scaler chỉ fit trên tập train, transform cả train và test
           → tránh data leakage
        │
        ▼
  [STEP 3] SMOTE (chỉ áp dụng trên tập train, sau khi split)
           Oversampling class Fraud từ ~0.17% lên cân bằng với class Normal
        │
        ├─────────────────────────────────────────────┐
        ▼                                             │
  [STEP 4] BASELINE                                   │
  Stacking (LR + SVM + KNN + XGBoost) trên            │
  FULL 30 features → ghi lại F1, ROC-AUC,             │  So sánh
  PR-AUC, Training Time cho từng model                │  Before / After
        │                                             │
        ▼                                             │
  [STEP 5] Feature Selection (RFE)                    │
  RFE với n_features cố định (mặc định 20),           │
  estimator = Random Forest (dùng để đo importance)   │
        │                                             │
        ▼                                             │
  [STEP 6] AFTER FS                                   │
  Stacking (LR + SVM + KNN + XGBoost) trên            │
  features đã chọn ───────────────────────────────────┘
        │
        ▼
  [STEP 7] Đánh giá 7 metrics trên tập test
  Accuracy · Precision · Recall · F1 · ROC-AUC · PR-AUC · MCC
        │
        ▼
  [STEP 8] Xuất 12 biểu đồ (fig01 – fig12) + models.pkl + results.json
```

**Mô hình trong Stacking Ensemble:**
- **Base learners:** Logistic Regression, SVM (RBF kernel), KNN, XGBoost
- **Meta-learner:** Logistic Regression, huấn luyện trên meta-features sinh từ 4 base learners
- **Kỹ thuật:** Out-of-Fold (OOF) qua `cross_val_predict` (StratifiedKFold) để sinh meta-features cho tập train, tránh meta-learner overfit lên chính base models; sau đó base models được refit trên toàn bộ tập train để dùng cho inference
- Thời gian huấn luyện (`training_time`) của mỗi model được đo **riêng từng model** (OOF + refit), không dùng chung một con số cho cả cụm

---

## Cấu trúc project

```
DACN/
│
├── src/                             # Mã nguồn chính (module hóa)
│   ├── data/
│   │   └── loader.py                # Load CSV, subsample majority class, stratified split
│   ├── preprocessing/
│   │   └── pipeline.py              # Preprocessor (StandardScaler) + SmoteBalancer
│   ├── feature_selection/
│   │   └── rfecv_selector.py        # RFEFixedSelector — Recursive Feature Elimination
│   ├── models/
│   │   └── base_models.py           # Builder cho LR, SVM, KNN, XGBoost (n_jobs=1 cho Windows)
│   ├── ensemble/
│   │   └── stacking.py              # StackingEnsemble (OOF meta-features)
│   ├── evaluation/
│   │   └── metrics.py               # Acc, Prec, Rec, F1, Specificity, FPR, ROC-AUC, PR-AUC, MCC
│   └── visualization/
│       └── plots.py                 # Xuất 12 biểu đồ (fig01 – fig12)
│
├── configs/
│   └── config.yaml                  # Toàn bộ cấu hình pipeline
│
├── data/                            # ← Đặt creditcard.csv ở đây (không commit vào git)
│
├── outputs/                         # Kết quả tự động sinh: biểu đồ, models.pkl, results.json
│
├── train.py                         # Entry point — chạy toàn bộ pipeline
├── requirements.txt
└── .gitignore                       # Bỏ qua data/, *.pkl, *.bat, *.json, __pycache__/
```

---

## Cài đặt & Chạy

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Tải dataset từ Kaggle và đặt vào thư mục data/
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    → data/creditcard.csv

# 3. Chạy pipeline với config mặc định
python train.py

# Tùy chọn: chỉ định config hoặc data path khác
python train.py --config configs/config.yaml
python train.py --data path/to/creditcard.csv
```

> **Windows:** code đã xử lý sẵn `n_jobs=1` cho các model dùng multiprocessing (KNN, XGBoost, Random Forest) và `if __name__ == '__main__':` guard trong `train.py` để tránh lỗi subprocess trên Windows.

---

## Cấu hình (`configs/config.yaml`)

```yaml
data:
  path: "data/creditcard.csv"
  test_size: 0.20              # 80/20 split
  random_state: 42
  majority_subsample: 50000    # Lấy mẫu con class đa số trước khi split (tăng tốc SMOTE)

feature_selection:
  n_features: 20                # Số features giữ lại
  estimator: "random_forest"    # Estimator dùng để đo feature importance cho RFE
  random_state: 42

models:
  logistic_regression: { C: 1.0, max_iter: 1000 }
  svm:                 { kernel: "rbf", C: 1.0 }
  knn:                 { n_neighbors: 5 }
  xgboost:              { n_estimators: 100, max_depth: 4, learning_rate: 0.1 }

stacking:
  cv_folds: 5                   # Số folds cho Out-of-Fold meta-feature generation

output:
  dir: "outputs"
  save_models: true
  save_plots: true
```

---

## Feature Selection: RFE-Fixed

`src/feature_selection/rfecv_selector.py` triển khai **`RFEFixedSelector`**:

| Thuộc tính | Mô tả |
|---|---|
| **Cách chọn** | RFE (Recursive Feature Elimination) với số features **cố định** (`n_features`, mặc định 20) |
| **Estimator** | `random_forest` (mặc định trong config) hoặc `xgboost` — dùng `feature_importances_` để loại dần feature yếu nhất |
| **Tốc độ** | Nhanh, không cần cross-validation lặp lại |
| **Kiểm soát** | Chính xác số features muốn giữ — phù hợp câu hỏi nghiên cứu dạng "15 hay 20 features tốt hơn?" |
| **Quy trình** | (1) Fit estimator → lấy `feature_importances_`; (2) loại bỏ feature ít quan trọng nhất; (3) lặp lại đến khi còn `n_features` features |

---

## Outputs (thư mục `outputs/`)

### Biểu đồ

| File | Nội dung |
|------|----------|
| `fig01_class_distribution.png` | Phân phối class trước và sau SMOTE |
| `fig03_feature_ranking.png` | Ranking toàn bộ features: selected (xanh) vs eliminated (xám) |
| `fig04_feature_importance.png` | Feature importance từ estimator, trên toàn bộ 30 features |
| `fig05_model_comparison.png` | So sánh 5 models × 7 metrics sau Feature Selection |
| `fig06_confusion_matrices.png` | Confusion matrices của từng model |
| `fig07_roc_curves.png` | ROC curves (AUC) của tất cả models |
| `fig08_pr_curves.png` | Precision-Recall curves |
| `fig09_metrics_heatmap.png` | Heatmap tổng hợp tất cả metrics |
| `fig10_before_after_comparison.png` | F1 / ROC-AUC / PR-AUC: Before vs After Feature Selection, có annotate delta (Δ) |
| `fig11_training_time_comparison.png` | Thời gian huấn luyện Before vs After FS theo từng model |
| `fig12_top_features_selected.png` | Top features được chọn, sắp xếp theo importance giảm dần |

### File dữ liệu

| File | Nội dung |
|------|----------|
| `models.pkl` | Object đã train: `preprocessor`, `selector`, `stacking`, `feature_names`, `selected_features` |
| `results.json` | Toàn bộ metrics của từng model (dùng để viết báo cáo hoặc phân tích thêm) |

> Cả hai file này bị `.gitignore` bỏ qua (`*.pkl`, `*.json`) — chúng chỉ tồn tại cục bộ sau khi chạy `train.py`, không nằm trong repo.

---

## Lý do chọn kỹ thuật

| Kỹ thuật | Lý do |
|----------|-------|
| **StandardScaler** | Chuẩn hóa input — cần thiết cho SVM (kernel RBF) và Logistic Regression |
| **Split trước SMOTE** | Tránh data leakage; test set giữ nguyên phân phối gốc để đánh giá thực tế |
| **SMOTE (chỉ trên train)** | Oversampling class thiểu số (Fraud ≈ 0.17%) — giữ nhiều thông tin hơn undersampling |
| **RFE-Fixed** | Chọn đúng N features cần thiết, nhanh, dễ giải thích và so sánh trong báo cáo |
| **Stacking (OOF)** | Kết hợp LR + SVM + KNN + XGBoost; kỹ thuật Out-of-Fold ngăn meta-learner overfit lên predictions của chính base models |
| **PR-AUC** | Quan trọng hơn ROC-AUC khi dữ liệu mất cân bằng nghiêm trọng (Fraud << Normal) |
| **MCC** | Metric toàn diện nhất — tính đến cả 4 ô của confusion matrix, không bị bias bởi class imbalance |

---

## Dataset

**Credit Card Fraud Detection** — Kaggle (ULB Machine Learning Group)

- Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284.807 giao dịch, 492 gian lận (~0.172%)
- 30 features: `V1`–`V28` (đã qua PCA), `Time`, `Amount`; nhãn `Class` (0 = Normal, 1 = Fraud)

Sau khi tải về, đặt file `creditcard.csv` vào thư mục `data/`.