"""
src/visualization/plots.py

  Fig 01 – Class Distribution (before / after SMOTE)
  Fig 02 – RFE/RFECV: CV Score vs Number of Features  (chỉ có với RFECV)
  Fig 03 – Feature Ranking (all features, selected vs eliminated)
  Fig 04 – Feature Importance (sorted, top-N highlight)
  Fig 05 – Model Comparison – Grouped Bar Chart (7 metrics)
  Fig 06 – Confusion Matrices Grid
  Fig 07 – ROC Curves (all models)
  Fig 08 – Precision-Recall Curves (all models)
  Fig 09 – Metrics Heatmap
  Fig 10 – Before vs After Feature Selection: F1 / ROC-AUC / PR-AUC per model
  Fig 11 – Training Time Comparison: Before vs After FS
  Fig 12 – Top Features Selected (importance bar chart, focus on selected only)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# ── Style chung ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.titlesize':  12,
    'axes.labelsize':  10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi':      120,
})

C_FRAUD   = '#E74C3C'
C_NORMAL  = '#27AE60'
C_SELECT  = '#2980B9'
C_ELIM    = '#BDC3C7'
C_BEFORE  = '#7F8C8D'   # màu "before FS"
C_AFTER   = '#E74C3C'   # màu "after FS"

MODEL_COLORS = {
    'Logistic Regression': '#3498DB',
    'SVM':                 '#E67E22',
    'KNN':                 '#9B59B6',
    'XGBoost':             '#F39C12',
    'Stacking Ensemble':   '#E74C3C',
}
FALLBACK = ['#3498DB','#E67E22','#9B59B6','#27AE60','#F39C12','#E74C3C']

def _color(name, idx=0):
    return MODEL_COLORS.get(name, FALLBACK[idx % len(FALLBACK)])

def _save(fig, path, dpi=150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Saved: {os.path.basename(path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 01: Class Distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_class_distribution(y_before, y_after, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Class Distribution: Before vs After SMOTE',
                 fontsize=14, fontweight='bold', y=1.02)
    for ax, y, title in [
        (axes[0], y_before, 'Before SMOTE (Original)'),
        (axes[1], y_after,  'After SMOTE  (Train set only)'),
    ]:
        counts = np.bincount(y.astype(int))
        labels = ['Normal (0)', 'Fraud (1)']
        bars = ax.bar(labels, counts, color=[C_NORMAL, C_FRAUD],
                      edgecolor='white', linewidth=1.5, width=0.5, alpha=0.9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Sample Count')
        ax.set_ylim(0, max(counts) * 1.18)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        total = sum(counts)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.02,
                    f'{cnt:,}\n({cnt/total*100:.2f}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    _save(fig, os.path.join(out_dir, 'fig01_class_distribution.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 02: RFECV CV Score (chỉ có với RFECV mode)
# ─────────────────────────────────────────────────────────────────────────────
def plot_rfecv_cv_score(selector, out_dir):
    cv_scores = selector.get_cv_scores()
    if cv_scores is None:
        print("  [skip fig02] RFE-Fixed không có CV scores.")
        return

    rfecv       = selector.rfecv_
    mean_scores = cv_scores
    std_scores  = selector.get_std_scores()
    min_f       = rfecv.min_features_to_select
    n_total     = len(rfecv.support_)
    optimal_n   = rfecv.n_features_
    x           = list(range(min_f, n_total + 1))[:len(mean_scores)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mean_scores, 'b-o', lw=2, ms=4,
            label=f'Mean CV Score ({rfecv.scoring})', zorder=3)
    ax.fill_between(x, mean_scores - std_scores, mean_scores + std_scores,
                    alpha=0.15, color='blue', label='± 1 std')
    opt_idx   = min(optimal_n - min_f, len(mean_scores) - 1)
    opt_score = mean_scores[opt_idx]
    ax.axvline(optimal_n, color='red', lw=2, ls='--',
               label=f'Optimal n = {optimal_n}', zorder=4)
    ax.scatter([optimal_n], [opt_score], color='red', s=120, zorder=5, marker='*')
    ax.annotate(f'n={optimal_n}\nScore={opt_score:.4f}',
                xy=(optimal_n, opt_score),
                xytext=(optimal_n + max(1, n_total * 0.04), opt_score - 0.01),
                fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
    ax.set_xlabel('Number of Features Selected'); ax.set_ylabel(f'CV Score ({rfecv.scoring})')
    ax.set_title(f'RFECV: CV Score vs Number of Features\n'
                 f'(Estimator: {selector.estimator_name.upper()}, CV={selector.cv} folds)',
                 fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    _save(fig, os.path.join(out_dir, 'fig02_rfecv_cv_score.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 03: Feature Ranking
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_ranking(selector, out_dir):
    rfe_obj    = selector.rfe_ if hasattr(selector, 'rfe_') else selector.rfecv_
    rankings   = rfe_obj.ranking_
    feat_names = selector.feature_names_
    optimal_n  = (selector.n_features if hasattr(selector, 'n_features')
                  else rfe_obj.n_features_)

    sort_idx     = np.argsort(rankings)
    sorted_names = [feat_names[i] for i in sort_idx]
    sorted_ranks = [rankings[i]   for i in sort_idx]
    colors       = [C_SELECT if r == 1 else C_ELIM for r in sorted_ranks]

    fig, ax = plt.subplots(figsize=(9, max(5, len(feat_names) * 0.38)))
    ax.barh(np.arange(len(sorted_names)), sorted_ranks, color=colors,
            edgecolor='white', height=0.7, alpha=0.9)
    ax.set_yticks(np.arange(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('RFE Ranking (1 = selected)')
    ax.set_title(f'Feature Ranking by RFE\n'
                 f'({optimal_n} selected in blue, eliminated in gray)',
                 fontsize=12, fontweight='bold')
    ax.axvline(1.5, color='red', lw=1.2, ls='--', alpha=0.6)
    ax.grid(axis='x', alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=C_SELECT, label=f'Selected ({optimal_n})'),
        Patch(facecolor=C_ELIM,   label=f'Eliminated ({len(feat_names)-optimal_n})'),
    ], fontsize=9, loc='lower right')
    _save(fig, os.path.join(out_dir, 'fig03_feature_ranking.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 04: Feature Importance (all features, từ estimator)
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(selector, out_dir):
    rfe_obj = selector.rfe_ if hasattr(selector, 'rfe_') else selector.rfecv_
    estimator = rfe_obj.estimator_
    if not hasattr(estimator, 'feature_importances_'):
        print("  [skip fig04] Estimator không có feature_importances_")
        return

    feat_names   = selector.feature_names_
    selected_set = set(selector.selected_names_)

    # estimator.feature_importances_ có độ dài = n_selected (sklearn)
    # Map lại về full feature space qua support_
    support = rfe_obj.support_
    imp_full = np.zeros(len(feat_names))
    imp_full[support] = estimator.feature_importances_

    sort_idx     = np.argsort(imp_full)[::-1]
    sorted_names = [feat_names[i] for i in sort_idx]
    sorted_imp   = [imp_full[i]   for i in sort_idx]
    colors       = [C_SELECT if n in selected_set else C_ELIM for n in sorted_names]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(np.arange(len(sorted_names)), sorted_imp, color=colors,
           edgecolor='white', linewidth=0.5, alpha=0.9)
    ax.set_xticks(np.arange(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Feature Importance (Gini/Gain)')
    ax.set_title(f'Feature Importance from {selector.estimator_name.upper()}\n'
                 f'(Blue = RFE selected, Gray = eliminated)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=C_SELECT, label='Selected by RFE'),
        Patch(facecolor=C_ELIM,   label='Eliminated'),
    ], fontsize=9)
    _save(fig, os.path.join(out_dir, 'fig04_feature_importance.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 05: Model Comparison – Grouped Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(results, out_dir):
    metrics = ['acc', 'prec', 'rec', 'f1', 'roc_auc', 'pr_auc', 'mcc']
    labels  = ['Accuracy','Precision','Recall','F1-Score','ROC-AUC','PR-AUC','MCC']
    model_names = list(results.keys())
    n_m, n_met = len(model_names), len(metrics)
    x = np.arange(n_met); w = 0.80 / n_m

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, name in enumerate(model_names):
        vals = [float(results[name].get(m, 0)) for m in metrics]
        ec   = 'black' if name == 'Stacking Ensemble' else 'white'
        lw   = 1.5     if name == 'Stacking Ensemble' else 0.5
        ax.bar(x + i * w - (n_m-1)*w/2, vals, w,
               label=name, color=_color(name, i), alpha=0.88,
               edgecolor=ec, linewidth=lw)
    ax.set_xlabel('Metric'); ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison (After Feature Selection)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, loc='upper left', ncol=2, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    _save(fig, os.path.join(out_dir, 'fig05_model_comparison.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 06: Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────
def plot_all_confusion_matrices(results, out_dir):
    n = len(results); ncols = 3; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 4.8*nrows))
    fig.suptitle('Confusion Matrices – All Models (After Feature Selection)',
                 fontsize=14, fontweight='bold', y=1.01)
    axes_flat = axes.flatten() if n > 1 else [axes]
    class_labels = ['Normal', 'Fraud']; cmap = plt.cm.Blues
    for idx, (name, res) in enumerate(results.items()):
        ax = axes_flat[idx]; cm = res['cm']
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(2)); ax.set_yticks(np.arange(2))
        ax.set_xticklabels(class_labels, fontsize=9)
        ax.set_yticklabels(class_labels, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('Actual', fontsize=9)
        ax.set_title(name, fontsize=11, fontweight='bold')
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='white' if cm[i,j] > thresh else 'black')
        ax.set_xlabel(
            f"Predicted  |  Rec={res.get('rec',0)*100:.1f}%  "
            f"Prec={res.get('prec',0)*100:.1f}%  F1={res.get('f1',0)*100:.1f}%",
            fontsize=8)
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    _save(fig, os.path.join(out_dir, 'fig06_confusion_matrices.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 07: ROC Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(results, y_test, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random (AUC=0.50)')
    for i, (name, res) in enumerate(results.items()):
        prob = res.get('prob')
        if prob is None: continue
        fpr_, tpr_, _ = roc_curve(y_test, prob)
        roc = res.get('roc_auc', auc(fpr_, tpr_))
        lw  = 2.8 if name == 'Stacking Ensemble' else 1.8
        ax.plot(fpr_, tpr_, lw=lw, color=_color(name, i),
                label=f'{name}  (AUC = {roc:.4f})')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curves – All Models (After Feature Selection)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8.5, loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    _save(fig, os.path.join(out_dir, 'fig07_roc_curves.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 08: Precision-Recall Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_pr_curves(results, y_test, out_dir):
    baseline = float(y_test.mean())
    fig, ax  = plt.subplots(figsize=(8, 6.5))
    ax.axhline(baseline, color='k', lw=1, ls='--', alpha=0.5,
               label=f'No-skill (AP={baseline:.4f})')
    for i, (name, res) in enumerate(results.items()):
        prob = res.get('prob')
        if prob is None: continue
        prec_, rec_, _ = precision_recall_curve(y_test, prob)
        pr = res.get('pr_auc', auc(rec_, prec_))
        lw = 2.8 if name == 'Stacking Ensemble' else 1.8
        ax.plot(rec_, prec_, lw=lw, color=_color(name, i),
                label=f'{name}  (AP = {pr:.4f})')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel('Recall (Sensitivity)'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves – All Models (After Feature Selection)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    _save(fig, os.path.join(out_dir, 'fig08_pr_curves.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 09: Metrics Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_heatmap(results, out_dir):
    try:
        import seaborn as sns
    except ImportError:
        print("  [skip fig09] seaborn chưa cài."); return
    keys   = ['acc','prec','rec','f1','roc_auc','pr_auc','mcc']
    labels = ['Accuracy','Precision','Recall','F1','ROC-AUC','PR-AUC','MCC']
    names  = list(results.keys())
    data   = np.array([[float(results[n].get(k,0)) for k in keys] for n in names])
    fig, ax = plt.subplots(figsize=(12, max(4, len(names)*0.75+1.5)))
    sns.heatmap(data, annot=True, fmt='.4f', cmap='RdYlGn',
                xticklabels=labels, yticklabels=names, ax=ax,
                linewidths=0.5, vmin=0, vmax=1,
                annot_kws={'fontsize':9,'fontweight':'bold'},
                cbar_kws={'label':'Score'})
    ax.set_title('Model Performance Heatmap – After Feature Selection',
                 fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(axis='x', labelsize=10, rotation=0)
    ax.tick_params(axis='y', labelsize=10, rotation=0)
    _save(fig, os.path.join(out_dir, 'fig09_metrics_heatmap.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10: Before vs After Feature Selection – F1 / ROC-AUC / PR-AUC
# ─────────────────────────────────────────────────────────────────────────────
def plot_before_after_comparison(comparison, fs_mode, out_dir):
    """
    Grouped bar chart: với mỗi metric (F1, ROC-AUC, PR-AUC),
    hiển thị giá trị Before (màu xám) và After (màu đỏ) cho từng model.
    """
    model_names = list(comparison.keys())
    metrics_info = [
        ('f1_before',      'f1_after',      'F1-Score',  '#3498DB', '#E74C3C'),
        ('roc_auc_before', 'roc_auc_after', 'ROC-AUC',   '#27AE60', '#E67E22'),
        ('pr_auc_before',  'pr_auc_after',  'PR-AUC',    '#9B59B6', '#F39C12'),
    ]

    n_models  = len(model_names)
    n_metrics = len(metrics_info)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6), sharey=False)

    x   = np.arange(n_models)
    w   = 0.35

    for ax, (k_before, k_after, metric_name, c_before, c_after) in zip(axes, metrics_info):
        vals_before = [comparison[n][k_before] for n in model_names]
        vals_after  = [comparison[n][k_after]  for n in model_names]

        bars_b = ax.bar(x - w/2, vals_before, w, label='Before FS',
                        color=c_before, alpha=0.55, edgecolor='white', hatch='//')
        bars_a = ax.bar(x + w/2, vals_after,  w, label='After FS',
                        color=c_after,  alpha=0.85, edgecolor='white')

        # Annotate delta trên thanh After
        for xi, (vb, va) in enumerate(zip(vals_before, vals_after)):
            delta = va - vb
            sign  = '+' if delta >= 0 else ''
            color = '#27AE60' if delta >= 0 else '#E74C3C'
            ax.text(xi + w/2, va + 0.005, f'{sign}{delta:.3f}',
                    ha='center', va='bottom', fontsize=7, color=color, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Score')
        ax.set_title(f'{metric_name}\nBefore vs After {fs_mode.upper()}',
                     fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.12)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, ls='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle(f'Before vs After Feature Selection ({fs_mode.upper()})\n'
                 f'F1, ROC-AUC, PR-AUC per Model',
                 fontsize=13, fontweight='bold', y=1.02)
    _save(fig, os.path.join(out_dir, 'fig10_before_after_comparison.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 11: Training Time Comparison – Before vs After FS
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_time_comparison(comparison, fs_mode, out_dir):
    """
    Horizontal bar chart: training time before và after FS cho từng model.
    """
    model_names   = list(comparison.keys())
    times_before  = [comparison[n]['time_before'] for n in model_names]
    times_after   = [comparison[n]['time_after']  for n in model_names]

    y   = np.arange(len(model_names))
    h   = 0.35
    fig, ax = plt.subplots(figsize=(10, max(5, len(model_names) * 0.8 + 1)))

    ax.barh(y + h/2, times_before, h, label=f'Before FS (full features)',
            color='#7F8C8D', alpha=0.65, edgecolor='white')
    ax.barh(y - h/2, times_after, h, label=f'After {fs_mode.upper()} (selected features)',
            color='#E74C3C', alpha=0.85, edgecolor='white')

    # Annotate thời gian giảm
    for i, (tb, ta) in enumerate(zip(times_before, times_after)):
        delta_pct = (ta - tb) / (tb + 1e-9) * 100
        sign = '+' if delta_pct >= 0 else ''
        color = '#27AE60' if delta_pct < 0 else '#E74C3C'
        ax.text(max(tb, ta) + 0.05, y[i],
                f'{sign}{delta_pct:.1f}%',
                va='center', fontsize=8, color=color, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(model_names, fontsize=10)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_title(f'Training Time Comparison: Before vs After {fs_mode.upper()}\n'
                 f'(Negative % = faster after feature selection)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    _save(fig, os.path.join(out_dir, 'fig11_training_time_comparison.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 12: Top Features Selected (focus on selected features only)
# ─────────────────────────────────────────────────────────────────────────────
def plot_top_features(selector, out_dir):
    """
    Horizontal bar chart: chỉ các features được chọn, sort theo importance.
    Màu gradient theo importance value.
    """
    rfe_obj   = selector.rfe_ if hasattr(selector, 'rfe_') else selector.rfecv_
    estimator = rfe_obj.estimator_

    if not hasattr(estimator, 'feature_importances_'):
        print("  [skip fig12] Estimator không có feature_importances_")
        return

    sel_names = selector.selected_names_
    imp_vals  = estimator.feature_importances_  # length = n_selected

    # Sort theo importance giảm dần
    sort_idx     = np.argsort(imp_vals)[::-1]
    sorted_names = [sel_names[i] for i in sort_idx]
    sorted_imp   = [imp_vals[i]  for i in sort_idx]

    # Màu gradient: từ đỏ đậm (quan trọng nhất) → xanh nhạt
    norm   = plt.Normalize(vmin=min(sorted_imp), vmax=max(sorted_imp))
    cmap   = plt.cm.RdYlGn
    colors = [cmap(norm(v)) for v in sorted_imp]

    n = len(sorted_names)
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.45 + 1.5)))

    y_pos = np.arange(n)
    bars  = ax.barh(y_pos, sorted_imp, color=colors, edgecolor='white',
                    height=0.7, alpha=0.92)

    # Annotate giá trị + rank
    for i, (bar, imp) in enumerate(zip(bars, sorted_imp)):
        ax.text(bar.get_width() + max(sorted_imp) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.4f}', va='center', fontsize=9, fontweight='bold')
        ax.text(-max(sorted_imp) * 0.01, bar.get_y() + bar.get_height()/2,
                f'#{i+1}', va='center', ha='right', fontsize=8,
                color='#555555')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('Feature Importance (Gini/Gain)')
    ax.set_title(
        f'Top {n} Selected Features – {selector.estimator_name.upper()}\n'
        f'(Sorted by importance, gradient: high=green → low=red)',
        fontsize=12, fontweight='bold'
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Importance Score', fontsize=9)

    ax.set_xlim(-max(sorted_imp) * 0.05, max(sorted_imp) * 1.18)
    ax.grid(axis='x', alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    _save(fig, os.path.join(out_dir, 'fig12_top_features_selected.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def save_all_plots(
    results,
    y_test,
    y_train_before_smote,
    y_train_after_smote,
    selector,
    fs_mode,
    comparison,
    out_dir,
):
    """
    Vẽ và lưu tất cả 12 biểu đồ.

    Parameters
    ----------
    results              : dict – kết quả evaluate_all() sau FS
    y_test               : ndarray
    y_train_before_smote : ndarray
    y_train_after_smote  : ndarray
    selector             : RFEFixedSelector hoặc RFECVSelector đã fit
    fs_mode              : str – 'rfe_fixed' hoặc 'rfecv'
    comparison           : dict – {model_name: {f1_before, f1_after, ...}}
    out_dir              : str
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[Plots] Saving all charts to: {out_dir}/")

    plot_class_distribution(y_train_before_smote, y_train_after_smote, out_dir)  # 01
    plot_rfecv_cv_score(selector, out_dir)                                        # 02
    plot_feature_ranking(selector, out_dir)                                       # 03
    plot_feature_importance(selector, out_dir)                                    # 04
    plot_model_comparison(results, out_dir)                                       # 05
    plot_all_confusion_matrices(results, out_dir)                                 # 06
    plot_roc_curves(results, y_test, out_dir)                                     # 07
    plot_pr_curves(results, y_test, out_dir)                                      # 08
    plot_metrics_heatmap(results, out_dir)                                        # 09
    plot_before_after_comparison(comparison, fs_mode, out_dir)                   # 10
    plot_training_time_comparison(comparison, fs_mode, out_dir)                  # 11
    plot_top_features(selector, out_dir)                                          # 12

    total = sum(1 for f in os.listdir(out_dir) if f.endswith('.png'))
    print(f"[Plots] ✓ {total} charts saved to {out_dir}/")
