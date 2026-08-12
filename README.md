# Credit Card Fraud Detection

Pipeline học máy phát hiện gian lận thẻ tín dụng với **Hybrid Feature Selection** và **Stacking Ensemble Learning**.

## Pipeline

```
Credit Card Fraud Detection Dataset
        │
        ▼
  StandardScaler (fit chỉ trên train)
        │
        ▼
  Train / Test Split  (80% / 20%)
        │
        ▼
  SMOTE  (chỉ trên tập train – tránh data leakage)
        │
        ├──────────────────────────────────────┐
        ▼                                      │
  [BASELINE] Train models                      │
  trên FULL features (30 features)             │
  → Ghi lại F1, ROC-AUC, PR-AUC, Time         │
        │                                      │
        ▼                                      │
  Feature Selection                            │ So sánh
  ┌─────────────────────┐                      │ Before / After
  │ mode = rfe_fixed    │  → chọn 15/20 feat  │
  │ mode = rfecv        │  → tự động tối ưu   │
  └─────────────────────┘                      │
        │                                      │
        ▼                                      │
  [AFTER FS] Train models                      │
  trên SELECTED features                       │
  → F1, ROC-AUC, PR-AUC, Time ────────────────┘
        │
        ▼
  Stacking Ensemble (OOF)
  RF + XGBoost + SVM → Logistic Regression (meta-learner)
        │
        ▼
  Đánh giá: Accuracy, Precision, Recall, F1,
             ROC-AUC, PR-AUC, MCC
        │
        ▼
  12 Biểu đồ (fig01 – fig12)
```

## Cấu trúc project

```
fraud_detection/
├── src/
│   ├── data/
│   │   └── loader.py              # Load & split data
│   ├── preprocessing/
│   │   └── pipeline.py            # StandardScaler + SMOTE
│   ├── feature_selection/
│   │   └── rfecv_selector.py      # RFEFixedSelector + RFECVSelector
│   ├── models/
│   │   └── base_models.py         # LR, SVM, KNN, RF, XGBoost builders
│   ├── ensemble/
│   │   └── stacking.py            # Stacking: RF+XGB+SVM → LR (OOF)
│   ├── evaluation/
│   │   └── metrics.py             # Acc, Prec, Rec, F1, ROC-AUC, PR-AUC, MCC
│   └── visualization/
│       └── plots.py               # 12 biểu đồ (fig01 – fig12)
├── configs/
│   └── config.yaml                # Cấu hình pipeline
├── data/                          # Đặt creditcard.csv ở đây
├── outputs/                       # Kết quả, model, biểu đồ
├── train.py                       # Main script
├── check_env.py
├── requirements.txt
├── setup_windows.bat
└── run_train.bat
```

## Cài đặt & Chạy

```bash
# Bước 1 – Setup
pip install -r requirements.txt
python check_env.py

# Bước 2 – Download creditcard.csv từ Kaggle và copy vào data/
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# Bước 3 – Chạy
python train.py
```

Windows: double-click `setup_windows.bat` rồi `run_train.bat`.

## Feature Selection: RFE-Fixed vs RFECV

| | RFE-Fixed (`mode: rfe_fixed`) | RFECV (`mode: rfecv`) |
|---|---|---|
| **Cách chọn** | Cố định n_features (15 hoặc 20) | Tự động tìm n tối ưu qua CV |
| **Tốc độ** | Nhanh (không cần CV) | Chậm hơn (CV × n_features lần fit) |
| **Kiểm soát** | Chính xác số features muốn giữ | Phụ thuộc CV score |
| **Thesis** | Tốt cho RQ: "15 hay 20 feature nào tốt hơn?" | Tốt cho RQ: "Bao nhiêu feature là tối ưu?" |
| **Config key** | `n_features: 20` | `min_features: 5`, `scoring: f1` |

Thay đổi trong `configs/config.yaml`:
```yaml
feature_selection:
  mode: 'rfe_fixed'    # Đổi thành 'rfecv' nếu muốn tự động
  n_features: 20       # Thử 15 hoặc 20
  estimator: 'xgboost' # Hoặc 'random_forest'
```

## Outputs (thư mục `outputs/`)

| File | Nội dung |
|------|----------|
| `fig01_class_distribution.png` | Phân phối class trước/sau SMOTE |
| `fig02_rfecv_cv_score.png` | CV score theo số features *(chỉ mode rfecv)* |
| `fig03_feature_ranking.png` | Ranking tất cả features (selected vs eliminated) |
| `fig04_feature_importance.png` | Feature importance từ estimator (full view) |
| `fig05_model_comparison.png` | So sánh 6 models × 7 metrics (after FS) |
| `fig06_confusion_matrices.png` | Confusion matrices |
| `fig07_roc_curves.png` | ROC curves |
| `fig08_pr_curves.png` | Precision-Recall curves |
| `fig09_metrics_heatmap.png` | Heatmap tổng kết |
| **`fig10_before_after_comparison.png`** | **F1 / ROC-AUC / PR-AUC Before vs After FS** |
| **`fig11_training_time_comparison.png`** | **Training time Before vs After FS (mỗi model)** |
| **`fig12_top_features_selected.png`** | **Top features được chọn + importance gradient** |
| `models.pkl` | Models đã train (preprocessor, selector, lr, knn, stack) |
| `results.json` | Kết quả metrics (JSON) |

## Ý nghĩa 3 biểu đồ mới (fig10–12)

### Fig 10 – Before vs After Feature Selection
So sánh F1, ROC-AUC, PR-AUC của từng model trước và sau khi áp dụng RFE.
- Cột xám (gạch chéo) = Before FS (toàn bộ 30 features)
- Cột màu = After FS (15 hoặc 20 features được chọn)
- Delta (Δ) được ghi trực tiếp trên mỗi thanh After

### Fig 11 – Training Time Comparison
So sánh thời gian huấn luyện (giây) của từng model.
Chứng minh rằng Feature Selection giúp giảm thời gian training.

### Fig 12 – Top Features Selected
Horizontal bar chart chỉ gồm các features được chọn, sort theo importance giảm dần.
Màu gradient green (quan trọng nhất) → red (ít quan trọng nhất).
Rank (#1, #2 ...) và score được annotate trực tiếp.

## Lý do chọn kỹ thuật

| Kỹ thuật | Lý do |
|----------|-------|
| **StandardScaler** | Chuẩn hóa input – cần thiết cho SVM và LR |
| **Split trước SMOTE** | Tránh data leakage – test set phản ánh distribution thực |
| **SMOTE** | Oversampling class minority (Fraud ~0.17%) |
| **RFE-Fixed** | Chọn đúng N features cần, nhanh, dễ giải thích trong thesis |
| **RFECV** | Tự động tìm N tối ưu – phù hợp phân tích chi tiết hơn |
| **Stacking (OOF)** | Kết hợp RF+XGB+SVM, OOF tránh overfitting meta-learner |
| **PR-AUC** | Quan trọng hơn ROC-AUC với imbalanced data |
| **MCC** | Metric toàn diện, không bị bias bởi class imbalance |
