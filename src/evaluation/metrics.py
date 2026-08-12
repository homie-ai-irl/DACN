"""
src/evaluation/metrics.py
Evaluation metrics đầy đủ:
  - Accuracy
  - Precision
  - Recall (Sensitivity / TPR) – quan trọng nhất với fraud detection
  - F1-Score
  - Specificity (TNR)
  - ROC-AUC  (Area Under ROC Curve)
  - PR-AUC   (Area Under Precision-Recall Curve = Average Precision)
  - MCC      (Matthews Correlation Coefficient) – metric tốt nhất cho imbalanced

Tại sao cần PR-AUC và MCC ngoài ROC-AUC?
  - ROC-AUC bị "optimistic" trên imbalanced dataset vì nó phụ thuộc TN (nhiều).
  - PR-AUC tập trung vào class minority (Fraud) → thực tế hơn.
  - MCC là metric duy nhất tính đến cả 4 ô confusion matrix → không bị bias.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,   # PR-AUC
    matthews_corrcoef,          # MCC
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Tính tất cả metrics cho một model.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like  (binary predictions)
    y_prob : array-like  (probability of positive class, optional)

    Returns
    -------
    dict với keys: acc, prec, rec, f1, spec, fpr, roc_auc, pr_auc, mcc, cm
    """
    cm            = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp + 1e-10)       # Specificity / TNR
    fpr  = fp / (fp + tn + 1e-10)       # False Positive Rate
    mcc  = matthews_corrcoef(y_true, y_pred)

    roc_auc = float('nan')
    pr_auc  = float('nan')
    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc  = average_precision_score(y_true, y_prob)

    return dict(
        acc=acc, prec=prec, rec=rec, f1=f1,
        spec=spec, fpr=fpr,
        roc_auc=roc_auc, pr_auc=pr_auc, mcc=mcc,
        cm=cm,
    )


def print_metrics_table(results: dict):
    """In bảng performance tương tự Table 4 trong paper."""
    W = 90
    header = (
        f"  {'Model':<22} "
        f"{'Acc%':>7} {'Prec%':>7} {'Rec%':>7} {'F1%':>7} "
        f"{'ROC-AUC':>8} {'PR-AUC':>8} {'MCC':>7}"
    )
    sep = "  " + "-" * (W - 2)

    print("\n" + "=" * W)
    print("  PERFORMANCE METRICS")
    print("=" * W)
    print(header)
    print(sep)
    for name, m in results.items():
        print(
            f"  {name:<22} "
            f"{m['acc']*100:>7.2f} "
            f"{m['prec']*100:>7.2f} "
            f"{m['rec']*100:>7.2f} "
            f"{m['f1']*100:>7.2f} "
            f"{m['roc_auc']:>8.4f} "
            f"{m['pr_auc']:>8.4f} "
            f"{m['mcc']:>7.4f}"
        )
    print(sep)
    print()


def evaluate_all(model_outputs: dict, y_true) -> dict:
    """
    Evaluate tất cả models.

    Parameters
    ----------
    model_outputs : dict  { model_name: {'pred': array, 'prob': array} }
    y_true        : array

    Returns
    -------
    dict { model_name: metrics_dict }
    """
    results = {}
    for name, out in model_outputs.items():
        m = compute_metrics(y_true, out['pred'], out.get('prob'))
        results[name] = m
    print_metrics_table(results)
    return results
