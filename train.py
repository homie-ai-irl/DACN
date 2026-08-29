"""
train.py – Main entry point
Pipeline:
  Data → StandardScaler → Split → SMOTE
       → [Baseline: Stacking(LR+SVM+KNN+XGB) trên full features]
       → RFE/RFECV Feature Selection
       → [After FS: Stacking(LR+SVM+KNN+XGB) trên selected features]
       → So sánh Before / After FS: F1, ROC-AUC, PR-AUC, Training Time
       → Biểu đồ: Feature Importance, Top Features, Comparison

Usage:
  python train.py
  python train.py --config configs/config.yaml
  python train.py --data path/to/creditcard.csv
"""

if __name__ == '__main__':
    import os, sys, argparse, time, pickle, json, warnings
    warnings.filterwarnings('ignore')

    import numpy as np
    import pandas as pd
    import yaml
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # ── Parse args ────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection Pipeline')
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--data',   default=None)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if args.data:
        cfg['data']['path'] = args.data

    OUT_DIR = cfg['output']['dir']
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    fs_mode = cfg['feature_selection'].get('mode', 'rfe_fixed')

    print("=" * 70)
    print("  Credit Card Fraud Detection Pipeline")
    print(f"  Base models : LR + SVM + KNN + XGBoost → Stacking Ensemble")
    print(f"  Mode        : StandardScaler → Split → SMOTE → {fs_mode.upper()} → Stacking")
    print("=" * 70)

    t_pipeline_start = time.time()

    # ── STEP 1: Load data ─────────────────────────────────────────────────────
    print("\n[STEP 1/8] Loading data ...")
    from src.data.loader import load_data, subsample_majority, split_data

    X, y, feature_names = load_data(cfg['data']['path'])
    majority_n = cfg['data'].get('majority_subsample', None)
    X, y = subsample_majority(X, y, majority_n, random_state=cfg['data']['random_state'])

    # ── STEP 2: Scale + Split ─────────────────────────────────────────────────
    print("\n[STEP 2/8] StandardScaler + Train/Test split ...")
    from src.preprocessing.pipeline import Preprocessor, SmoteBalancer

    X_train_raw, X_test_raw, y_train, y_test = split_data(
        X, y,
        test_size    = cfg['data']['test_size'],
        random_state = cfg['data']['random_state'],
    )
    preprocessor   = Preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train_raw)
    X_test_scaled  = preprocessor.transform(X_test_raw)
    print(f"  → Train: {X_train_scaled.shape}  |  Test: {X_test_scaled.shape}")

    # ── STEP 3: SMOTE ─────────────────────────────────────────────────────────
    print("\n[STEP 3/8] SMOTE balancing (train only) ...")
    y_train_before_smote = y_train.copy()
    balancer = SmoteBalancer(random_state=cfg['preprocessing']['random_state'])
    X_train_bal, y_train_bal = balancer.fit_resample(X_train_scaled, y_train)
    print(f"  → {X_train_scaled.shape[0]} → {X_train_bal.shape[0]} samples after SMOTE")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: BASELINE – Stacking(LR+SVM+KNN+XGB) trên FULL features
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 4/8] BASELINE training (LR + SVM + KNN + XGBoost, full features) ...")
    from src.ensemble.stacking import StackingEnsemble
    from src.evaluation.metrics import compute_metrics

    t0 = time.time()
    stack_base = StackingEnsemble(cfg)
    stack_base.fit(X_train_bal, y_train_bal)
    stack_time_before = time.time() - t0

    stk_base_out = stack_base.predict_all_models(X_test_scaled)
    baseline_results = {}
    for model_name in ['Logistic Regression', 'SVM', 'KNN', 'XGBoost', 'Stacking Ensemble']:
        o = stk_base_out.get(model_name, {})
        baseline_results[model_name] = {
            **compute_metrics(y_test, o['pred'], o.get('prob')),
            # Thời gian huấn luyện RIÊNG từng model (không còn gán chung
            # stack_time_before cho cả 5 dòng — xem stacking.py:model_times_)
            'training_time': stack_base.model_times_[model_name],
        }

    print(f"\n  Baseline results (full {X_train_bal.shape[1]} features):")
    for nm, m in baseline_results.items():
        print(f"    {nm:<22} F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}  "
              f"PR-AUC={m['pr_auc']:.4f}  t={m['training_time']:.2f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Feature Selection
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[STEP 5/8] Feature Selection (mode={fs_mode.upper()}) ...")
    from src.feature_selection.rfecv_selector import RFEFixedSelector

    fs_cfg = cfg['feature_selection']

    if fs_mode == 'rfe_fixed':
        selector = RFEFixedSelector(
            n_features   = fs_cfg.get('n_features', 20),
            estimator    = fs_cfg.get('estimator', 'xgboost'),
            random_state = fs_cfg.get('random_state', 42),
        )
    else:  # 'rfecv'
        selector = RFECVSelector(
            estimator    = fs_cfg.get('estimator', 'xgboost'),
            cv           = fs_cfg.get('cv', 5),
            scoring      = fs_cfg.get('scoring', 'f1'),
            min_features = fs_cfg.get('min_features', 5),
            random_state = fs_cfg.get('random_state', 42),
        )

    selector.fit(X_train_bal, y_train_bal, feature_names)
    X_train_sel = selector.transform(X_train_bal)
    X_test_sel  = selector.transform(X_test_scaled)
    n_sel       = X_train_sel.shape[1]
    print(f"  → {X_train_bal.shape[1]} → {n_sel} features selected")
    print(f"  → Selected: {selector.get_selected_names()}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: AFTER FS – Stacking(LR+SVM+KNN+XGB) trên selected features
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[STEP 6/8] Stacking Ensemble on {n_sel} selected features ...")
    t0 = time.time()
    stack = StackingEnsemble(cfg)
    stack.fit(X_train_sel, y_train_bal)
    stack_time_after = time.time() - t0

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Evaluate AFTER FS – tất cả models
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[STEP 7/8] Evaluating all models (after FS) ...")
    from src.evaluation.metrics import evaluate_all

    all_outputs = stack.predict_all_models(X_test_sel)
    after_results = evaluate_all(all_outputs, y_test)

    # Gắn prob + training_time (riêng từng model) vào after_results
    for name in after_results:
        after_results[name]['prob']          = all_outputs[name].get('prob')
        after_results[name]['training_time'] = stack.model_times_[name]

    # Xây dựng comparison dict
    comparison = {
        name: {
            'f1_before':       baseline_results[name]['f1'],
            'f1_after':        after_results[name]['f1'],
            'roc_auc_before':  baseline_results[name]['roc_auc'],
            'roc_auc_after':   after_results[name]['roc_auc'],
            'pr_auc_before':   baseline_results[name]['pr_auc'],
            'pr_auc_after':    after_results[name]['pr_auc'],
            'time_before':     baseline_results[name]['training_time'],
            'time_after':      after_results[name]['training_time'],
        }
        for name in after_results
        if name in baseline_results
    }

    # In bảng so sánh
    from src.evaluation.metrics import print_metrics_table
    print(f"\n── COMPARISON Before vs After {fs_mode.upper()} (n={n_sel}) ─────────────")
    print(f"  {'Model':<22}  {'F1 Δ':>8}  {'ROC Δ':>8}  {'PR-AUC Δ':>9}  {'Time Δ':>8}")
    print("  " + "-" * 62)
    for nm, c in comparison.items():
        d_f1  = c['f1_after']      - c['f1_before']
        d_roc = c['roc_auc_after'] - c['roc_auc_before']
        d_pr  = c['pr_auc_after']  - c['pr_auc_before']
        d_t   = c['time_after']    - c['time_before']
        sign  = lambda v: '+' if v >= 0 else ''
        print(f"  {nm:<22}  {sign(d_f1)}{d_f1:+.4f}  {sign(d_roc)}{d_roc:+.4f}"
              f"  {sign(d_pr)}{d_pr:+.4f}    {d_t:+.2f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8: Save charts & models
    # ─────────────────────────────────────────────────────────────────────────
    if cfg['output'].get('save_plots', True):
        from src.visualization.plots import save_all_plots
        save_all_plots(
            results              = after_results,
            y_test               = y_test,
            y_train_before_smote = y_train_before_smote,
            y_train_after_smote  = y_train_bal,
            selector             = selector,
            fs_mode              = fs_mode,
            comparison           = comparison,
            out_dir              = OUT_DIR,
        )

    if cfg['output'].get('save_models', True):
        print("\n[Save] Saving models ...")
        model_path = os.path.join(OUT_DIR, 'models.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'preprocessor':      preprocessor,
                'selector':          selector,
                'stacking':          stack,
                'feature_names':     feature_names,
                'selected_features': selector.get_selected_names(),
            }, f)
        print(f"  ✓ {model_path}")

        results_json = {}
        for nm, m in after_results.items():
            results_json[nm] = {
                k: float(v) for k, v in m.items()
                if k not in ('cm', 'prob')
            }
            results_json[nm]['confusion_matrix'] = m['cm'].tolist()
        with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"  ✓ {OUT_DIR}/results.json")

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_pipeline_start
    print(f"\n{'='*70}")
    
    print(f"  PIPELINE COMPLETE  ({elapsed:.1f}s total)")
    print(f"  Mode: {fs_mode.upper()}  |  Selected features ({n_sel}): {selector.get_selected_names()}")
    print(f"  Outputs: {OUT_DIR}/")
    print(f"{'='*70}")