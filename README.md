# Credit Card Fraud Detection

Hệ thống phát hiện gian lận thẻ tín dụng sử dụng **Hybrid Feature Selection** (RFE-Fixed / RFECV) kết hợp **Stacking Ensemble Learning** (Out-of-Fold). Dự án được xây dựng phục vụ nghiên cứu khoa học (luận văn tốt nghiệp), tập trung chứng minh hai luận điểm:

1. **Feature Selection** (RFE) giúp cải thiện hiệu năng phát hiện gian lận và giảm thời gian huấn luyện so với dùng toàn bộ 30 features.
2. **Stacking Ensemble** (LR + SVM + KNN + XGBoost → meta-learner LR) vượt trội hơn từng mô hình đơn lẻ trên dữ liệu mất cân bằng nghiêm trọng (Fraud ≈ 0.17%).

---

## Mục lục

- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Pipeline tổng quan](#pipeline-tổng-quan)
- [Cấu trúc project](#cấu-trúc-project)
- [Cài đặt & Chạy](#cài-đặt--chạy)
- [Cấu hình](#cấu-hình-configsconfyaml)
- [Feature Selection: RFE-Fixed vs RFECV](#feature-selection-rfe-fixed-vs-rfecv)
- [Outputs](#outputs-thư-mục-outputs)
- [Lý do chọn kỹ thuật](#lý-do-chọn-kỹ-thuật)

---

## Yêu cầu hệ thống

| Thành phần     | Phiên bản tối thiểu |
|----------------|----------------------|
| Python         | 3.10+                |
| numpy          | ≥ 1.26.0             |
| pandas         | ≥ 2.1.0              |
| scikit-learn   | ≥ 1.4.0              |
| xgboost        | ≥ 2.0.0              |
| imbalanced-learn | ≥ 0.12.0           |
| matplotlib     | ≥ 3.8.0              |
| seaborn        | ≥ 0.13.0             |
| pyyaml         | ≥ 6.0                |
| Pillow         | ≥ 10.0.0             |

---

## Pipeline tổng quan

```
Credit Card Fraud Detection Dataset (creditcard.csv)
        │
        ▼
  [STEP 1] Load Data
  StandardScaler fit chỉ trên tập train → tránh data leakage
        │
        ▼
  [STEP 2] Train / Test Split  (80% / 20%, stratified)
        │
        ▼
  [STEP 3] SMOTE  (chỉ áp dụng trên tập train)
  Oversampling class Fraud từ ~0.17% → cân bằng
        │
        ├───────────────────────────────────────────┐
        ▼                                           │
  [STEP 4] BASELINE                                 │
  Stacking(LR + SVM + KNN + XGBoost) trên           │
  FULL 30 features → ghi lại F1, ROC-AUC,           │ So sánh
  PR-AUC, Training Time                             │ Before / After
        │                                           │
        ▼                                           │
  [STEP 5] Feature Selection (RFE)                  │
  ┌──────────────────────────────────┐              │
  │ mode = rfe_fixed  → chọn N feat  │              │
  │ mode = rfecv      → tự động CV   │              │
  └──────────────────────────────────┘              │
        │                                           │
        ▼                                           │
  [STEP 6] AFTER FS                                 │
  Stacking(LR + SVM + KNN + XGBoost) trên           │
  SELECTED features → F1, ROC-AUC, PR-AUC ─────────┘
        │
        ▼
  [STEP 7] Đánh giá 7 metrics
  Accuracy · Precision · Recall · F1
  ROC-AUC · PR-AUC · MCC
        │
        ▼
  [STEP 8] Xuất 12 biểu đồ (fig01 – fig12)
```

**Mô hình trong Stacking Ensemble:**
- **Base learners:** Logistic Regression, SVM (RBF), KNN, XGBoost
- **Meta-learner:** Logistic Regression
- **Phương pháp:** Out-of-Fold (OOF) để tránh overfitting meta-learner

---

## Cấu trúc project

```
fraud_detection/
│
├── src/                            # Mã nguồn chính (module hóa)
│   ├── data/
│   │   └── loader.py               # Load, subsample, split dataset
│   ├── preprocessing/
│   │   └── pipeline.py             # StandardScaler + SMOTE
│   ├── feature_selection/
│   │   └── rfecv_selector.py       # RFEFixedSelector + RFECVSelector
│   ├── models/
│   │   └── base_models.py          # Builder cho LR, SVM, KNN, XGBoost
│   ├── ensemble/
│   │   └── stacking.py             # Stacking Ensemble (OOF)
│   ├── evaluation/
│   │   └── metrics.py              # Acc, Prec, Rec, F1, ROC-AUC, PR-AUC, MCC
│   └── visualization/
│       └── plots.py                # Xuất 12 biểu đồ (fig01 – fig12)
│
├── configs/
│   └── config.yaml                 # Toàn bộ cấu hình pipeline
│
├── data/                           # ← Đặt creditcard.csv ở đây
│
├── outputs/                        # Kết quả tự động sinh: model, JSON, biểu đồ
│
├── train.py                        # Entry point – chạy toàn bộ pipeline
├── requirements.txt

```

---

## Cài đặt & Chạy

### Linux / macOS

```bash
# 1. Cài dependencies
pip install -r requirements.txt


# 2. Tải dataset từ Kaggle và đặt vào thư mục data/
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    → data/creditcard.csv

# 3. Chạy pipeline
python train.py

# Tùy chọn: chỉ định config hoặc data path
python train.py --config configs/config.yaml
python train.py --data path/to/creditcard.csv
```

## Cấu hình (`configs/config.yaml`)

```yaml
data:
  path: "data/creditcard.csv"
  test_size: 0.20          # 80/20 split
  random_state: 42
  majority_subsample: 50000  # Lấy mẫu con từ class đa số (tăng tốc)

feature_selection:
  mode: 'rfe_fixed'        # Đổi thành 'rfecv' để tự động chọn n
  n_features: 20           # Số features giữ lại (chỉ dùng với rfe_fixed)
  estimator: "random_forest"   # Estimator cho RFE: "xgboost" hoặc "random_forest"
  cv: 5                    # Số folds CV (chỉ dùng với rfecv)
  scoring: "f1"            # Metric tối ưu: "f1", "roc_auc", "average_precision"
  min_features: 5          # Số features tối thiểu (chỉ dùng với rfecv)

models:
  logistic_regression: { C: 1.0, max_iter: 1000 }
  svm:                 { kernel: "rbf", C: 1.0 }
  knn:                 { n_neighbors: 5 }
  xgboost:             { n_estimators: 100, max_depth: 4, learning_rate: 0.1 }

stacking:
  cv_folds: 5
```

---

## Feature Selection: RFE-Fixed vs RFECV

| | `mode: rfe_fixed` | `mode: rfecv` |
|---|---|---|
| **Cách chọn** | Cố định N features (15 hoặc 20) | Tự động tìm N tối ưu qua Cross-Validation |
| **Tốc độ** | Nhanh (không cần CV lặp) | Chậm hơn (CV × n_features lần fit) |
| **Kiểm soát** | Chính xác số features muốn giữ | Phụ thuộc CV score |
| **Phù hợp** | RQ: "15 hay 20 features, cái nào tốt hơn?" | RQ: "Bao nhiêu features là tối ưu?" |
| **Config** | `n_features: 20` | `min_features: 5`, `scoring: f1` |

**Đổi chế độ** trong `configs/config.yaml`:

```yaml
feature_selection:
  mode: 'rfecv'        # hoặc 'rfe_fixed'
  n_features: 20       # chỉ dùng với rfe_fixed; thử 15 hoặc 20
  estimator: 'xgboost' # hoặc 'random_forest'
```

---

## Outputs (thư mục `outputs/`)

### Biểu đồ

| File | Nội dung |
|------|----------|
| `fig01_class_distribution.png` | Phân phối class trước và sau SMOTE |
| `fig02_rfecv_cv_score.png` | CV score theo số features *(chỉ xuất ở mode `rfecv`)* |
| `fig03_feature_ranking.png` | Ranking toàn bộ features: selected vs eliminated |
| `fig04_feature_importance.png` | Feature importance từ estimator (toàn bộ 30 features) |
| `fig05_model_comparison.png` | So sánh 5 models × 7 metrics sau Feature Selection |
| `fig06_confusion_matrices.png` | Confusion matrices của từng model |
| `fig07_roc_curves.png` | ROC curves (AUC) |
| `fig08_pr_curves.png` | Precision-Recall curves |
| `fig09_metrics_heatmap.png` | Heatmap tổng kết tất cả metrics |
| **`fig10_before_after_comparison.png`** | **F1 / ROC-AUC / PR-AUC: Before vs After Feature Selection** |
| **`fig11_training_time_comparison.png`** | **Thời gian huấn luyện Before vs After FS (theo từng model)** |
| **`fig12_top_features_selected.png`** | **Top features được chọn, sort theo importance (gradient màu)** |

> **Fig 10 – Before vs After FS:** Cột xám (gạch chéo) = 30 features (Baseline); Cột màu = features được chọn (After FS). Delta (Δ) được annotate trực tiếp trên từng cột After, chứng minh mức cải thiện.
>
> **Fig 11 – Training Time:** Chứng minh Feature Selection giảm thời gian huấn luyện, đặc biệt với SVM và KNN nhạy cảm với số chiều.
>
> **Fig 12 – Top Features:** Horizontal bar chart chỉ hiển thị features được chọn, sắp xếp theo importance giảm dần với gradient màu từ xanh lá (quan trọng nhất) đến đỏ. Rank và score được annotate trực tiếp.

### File dữ liệu

| File | Nội dung |
|------|----------|
| `models.pkl` | Models đã train: preprocessor, selector, base models, stacking |
| `results.json` | Toàn bộ kết quả metrics (JSON, dùng để báo cáo hoặc phân tích thêm) |

---

## Lý do chọn kỹ thuật

| Kỹ thuật | Lý do |
|----------|-------|
| **StandardScaler** | Chuẩn hóa input – bắt buộc với SVM (kernel RBF) và Logistic Regression |
| **Split trước SMOTE** | Tránh data leakage; test set phản ánh đúng phân phối thực tế |
| **SMOTE** | Oversampling class thiểu số (Fraud ≈ 0.17%) – tốt hơn undersampling về thông tin giữ lại |
| **RFE-Fixed** | Chọn đúng N features cần thiết, nhanh, dễ giải thích trong báo cáo |
| **RFECV** | Tự động tìm N tối ưu qua CV – phù hợp phân tích thực nghiệm sâu hơn |
| **Stacking (OOF)** | Kết hợp LR + SVM + KNN + XGBoost; OOF ngăn meta-learner overfit trên predictions của base models |
| **PR-AUC** | Quan trọng hơn ROC-AUC với dữ liệu mất cân bằng nghiêm trọng (Fraud << Normal) |
| **MCC** | Metric toàn diện nhất – không bị bias bởi class imbalance, phù hợp khi TP/TN/FP/FN đều quan trọng |

---

## Dataset

**Credit Card Fraud Detection** – Kaggle (ULB Machine Learning Group)

- Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284,807 giao dịch · 492 gian lận (0.172%)
- 30 features: `V1`–`V28` (PCA), `Time`, `Amount`; nhãn `Class` (0 = Normal, 1 = Fraud)

Sau khi tải về, đặt file `creditcard.csv` vào thư mục `data/`.