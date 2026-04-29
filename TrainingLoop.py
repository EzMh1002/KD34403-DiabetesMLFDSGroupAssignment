import pandas as pd
import numpy as np
from sklearn.model_selection import (StratifiedKFold, learning_curve, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KD34403 - Group 13: Milestone 3 - The Training Loop
# Dataset: Pima Indians Diabetes Database (768 samples, 8 features)
# Model: Random Forest (selected in Milestone 2)
# Primary Metric: Recall (clinical priority — minimize missed diagnoses)
# Secondary Metrics: F1-Score, ROC AUC
# ============================================================

print("=" * 65)
print("  MILESTONE 3: THE TRAINING LOOP")
print("  KD34403 Group 13 - Pima Indians Diabetes Classification")
print("  Model: Random Forest | Primary Metric: Recall")
print("=" * 65)

# ============================================================
# STEP 1: LOAD CLEANED DATA (from Milestone 1 - DataCleaning.py)
# ============================================================
df = pd.read_csv('cleaned_diabetes_data.csv')

X = df.drop('target', axis=1)
y = df['target']

feature_names = X.columns.tolist()

print(f"\n[STEP 1] Data Loaded (from Milestone 1 pipeline)")
print(f"  Samples: {X.shape[0]}  |  Features: {X.shape[1]}")
print(f"  Features: {feature_names}")
print(f"\n  Class Distribution:")
print(f"    No Diabetes (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
print(f"    Diabetes (1):    {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
print(f"    ** Imbalanced dataset — Recall is critical to avoid missing positive cases **")

# ============================================================
# STEP 2: TRAIN / VALIDATION / TEST SPLIT (70 / 15 / 15)
# ============================================================
# Single stratified split into 70% train, 15% val, 15% test
np.random.seed(67)

# Sort indices by class to ensure stratification
idx_class0 = y[y == 0].index.tolist()
idx_class1 = y[y == 1].index.tolist()

np.random.shuffle(idx_class0)
np.random.shuffle(idx_class1)

# Calculate split points for each class (70/15/15)
def split_indices(indices, train_ratio=0.70, val_ratio=0.15):
    n = len(indices)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]

train_idx_0, val_idx_0, test_idx_0 = split_indices(idx_class0)
train_idx_1, val_idx_1, test_idx_1 = split_indices(idx_class1)

# Combine indices
train_idx = train_idx_0 + train_idx_1
val_idx = val_idx_0 + val_idx_1
test_idx = test_idx_0 + test_idx_1

# Create splits
X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_val, y_val = X.loc[val_idx], y.loc[val_idx]
X_test, y_test = X.loc[test_idx], y.loc[test_idx]

print(f"\n[STEP 2] Data Split — 70/15/15 (Stratified)")
print(f"  Training set:    {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Validation set:  {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"  Test set:        {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%) [HELD OUT for Milestone 5]")

print(f"\n  Class distribution preserved in each split:")
print(f"    Train  - Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")
print(f"    Val    - Class 0: {(y_val == 0).sum()}, Class 1: {(y_val == 1).sum()}")
print(f"    Test   - Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()}")

# ============================================================
# STEP 3: FEATURE SCALING
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\n[STEP 3] Feature Scaling Applied (StandardScaler)")
print(f"  Note: RF is tree-based and scale-invariant, but applied for pipeline consistency")
print(f"  Scaler fitted on training set ONLY (prevents data leakage)")

# ============================================================
# STEP 4: RANDOM FOREST TRAINING - MULTIPLE CONFIGURATIONS
# As justified in Milestone 2, Random Forest was selected for:
#   1. Superiority on structured tabular data
#   2. Strong generalization on small samples (n=768)
#   3. Non-linear feature interaction modeling
#   4. Native feature importance (clinical interpretability)
# ============================================================
print(f"\n[STEP 4] TRAINING RANDOM FOREST - MULTIPLE CONFIGURATIONS")
print("=" * 65)
print("  Testing different tree counts to observe training behavior...")

n_estimators_list = [10, 25, 50, 75, 100, 150, 200]
training_progress = []

for n_trees in n_estimators_list:
    rf = RandomForestClassifier(
        n_estimators=n_trees, 
        random_state=67,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    y_train_pred = rf.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    
    y_val_pred = rf.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    y_val_proba = rf.predict_proba(X_val_scaled)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    training_progress.append({
        'n_trees': n_trees,
        'train_acc': train_acc,
        'train_recall': train_recall,
        'val_acc': val_acc,
        'val_recall': val_recall,
        'val_precision': val_prec,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'overfit_gap': train_acc - val_acc
    })
    
    print(f"\n  n_estimators = {n_trees:>3}")
    print(f"    Train Acc: {train_acc:.4f} | Train Recall: {train_recall:.4f}")
    print(f"    Val Acc:   {val_acc:.4f} | Val Recall:   {val_recall:.4f} | Val F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
    print(f"    Overfit Gap: {train_acc - val_acc:.4f}  {'⚠️ OVERFITTING' if (train_acc - val_acc) > 0.10 else '✅ Acceptable'}")

progress_df = pd.DataFrame(training_progress)

# ============================================================
# STEP 5: DETAILED EVALUATION OF BEST CONFIGURATION
# ============================================================
best_idx = progress_df['val_recall'].idxmax()
best_config = progress_df.iloc[best_idx]

print(f"\n\n[STEP 5] BEST CONFIGURATION (by Validation Recall)")
print("=" * 65)
print(f"  Best n_estimators:   {int(best_config['n_trees'])}")
print(f"  Validation Recall:   {best_config['val_recall']:.4f}  << PRIMARY METRIC")
print(f"  Validation Accuracy: {best_config['val_acc']:.4f}")
print(f"  Validation F1-Score: {best_config['val_f1']:.4f}")
print(f"  Validation ROC AUC:  {best_config['val_auc']:.4f}")
print(f"  Overfit Gap:         {best_config['overfit_gap']:.4f}")

best_rf = RandomForestClassifier(
    n_estimators=int(best_config['n_trees']),
    random_state=67,
    n_jobs=-1
)
best_rf.fit(X_train_scaled, y_train)
y_val_pred_best = best_rf.predict(X_val_scaled)
y_val_proba_best = best_rf.predict_proba(X_val_scaled)[:, 1]

print(f"\n  Full Classification Report (Validation Set):")
print(classification_report(y_val, y_val_pred_best, 
                             target_names=['No Diabetes (0)', 'Diabetes (1)'],
                             digits=4))

# ============================================================
# STEP 6: 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================
print(f"\n[STEP 6] 5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=67)

cv_metrics = {
    'accuracy': 'accuracy',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'precision': 'precision'
}

# Combine train + val for cross-validation (exclude test set)
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

print(f"  Model: Random Forest (n_estimators={int(best_config['n_trees'])})")
print(f"  Cross-validation on train+val set ({X_trainval.shape[0]} samples)\n")

scaler_cv = StandardScaler()
X_trainval_scaled = scaler_cv.fit_transform(X_trainval)

cv_scores_stored = {}

for metric_name, scorer in cv_metrics.items():
    scores = cross_val_score(
        RandomForestClassifier(n_estimators=int(best_config['n_trees']), random_state=67),
        X_trainval_scaled, y_trainval, cv=cv, scoring=scorer
    )
    cv_scores_stored[metric_name] = scores  # ← THIS IS THE ONLY NEW LINE
    print(f"  {metric_name.upper():<12} | Mean: {scores.mean():.4f} | Std: {scores.std():.4f} | Folds: [{', '.join(f'{s:.4f}' for s in scores)}]")
# ============================================================
# STEP 7: VISUALIZATIONS
# ============================================================
print(f"\n\n[STEP 7] GENERATING TRAINING VISUALIZATIONS")
print("=" * 65)

# --- FIGURE 1: Training Progress (Accuracy & Recall vs n_estimators) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(progress_df['n_trees'], progress_df['train_acc'], 'o-', 
             color='#3498db', linewidth=2, markersize=8, label='Training Accuracy')
axes[0].plot(progress_df['n_trees'], progress_df['val_acc'], 's-', 
             color='#e74c3c', linewidth=2, markersize=8, label='Validation Accuracy')
axes[0].set_xlabel('Number of Trees (n_estimators)', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Training vs Validation Accuracy', fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0.65, 1.05)

axes[1].plot(progress_df['n_trees'], progress_df['train_recall'], 'o-', 
             color='#3498db', linewidth=2, markersize=8, label='Training Recall')
axes[1].plot(progress_df['n_trees'], progress_df['val_recall'], 's-', 
             color='#e74c3c', linewidth=2, markersize=8, label='Validation Recall')
axes[1].axhline(y=0.76, color='green', linestyle='--', linewidth=1.5, 
                label='ADAP Benchmark (76%)')
axes[1].set_xlabel('Number of Trees (n_estimators)', fontsize=12)
axes[1].set_ylabel('Recall', fontsize=12)
axes[1].set_title('Training vs Validation Recall (PRIMARY METRIC)', fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0.40, 1.05)

plt.suptitle('Milestone 3: Random Forest Training Progress\nGroup 13 - Pima Indians Diabetes', 
             fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [7a] Saved: training_progress.png")

# --- FIGURE 2: All Metrics vs n_estimators ---
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(progress_df['n_trees'], progress_df['val_acc'], 'o-', label='Accuracy', linewidth=2)
ax.plot(progress_df['n_trees'], progress_df['val_recall'], 's-', label='Recall ★', linewidth=2.5, color='#e74c3c')
ax.plot(progress_df['n_trees'], progress_df['val_precision'], '^-', label='Precision', linewidth=2)
ax.plot(progress_df['n_trees'], progress_df['val_f1'], 'D-', label='F1-Score', linewidth=2)
ax.plot(progress_df['n_trees'], progress_df['val_auc'], 'v-', label='ROC AUC', linewidth=2)

ax.axhline(y=0.76, color='green', linestyle='--', linewidth=1.5, label='ADAP Benchmark (76%)')
ax.set_xlabel('Number of Trees (n_estimators)', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Milestone 3: Validation Metrics vs Number of Trees\n★ Recall is Primary Metric (Milestone 2 Decision)', 
             fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.40, 1.05)

plt.tight_layout()
plt.savefig('validation_metrics_progression.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [7b] Saved: validation_metrics_progression.png")

# --- FIGURE 3: Learning Curve ---
fig, ax = plt.subplots(figsize=(10, 6))

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=int(best_config['n_trees']), random_state=67),
    X_trainval_scaled, y_trainval,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='recall', random_state=67
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

ax.plot(train_sizes, train_mean, 'o-', color='#3498db', linewidth=2, label='Training Recall')
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#3498db')
ax.plot(train_sizes, val_mean, 'o-', color='#e74c3c', linewidth=2, label='Cross-Val Recall')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='#e74c3c')

ax.set_xlabel('Number of Training Samples', fontsize=12)
ax.set_ylabel('Recall', fontsize=12)
ax.set_title('Milestone 3: Learning Curve (Random Forest)\nShows how model recall improves with more training data', 
             fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.50, 1.05)

plt.tight_layout()
plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [7c] Saved: learning_curve.png")

# --- FIGURE 4: Confusion Matrices ---
# --- FIGURE 4: Confusion Matrices (Textbook Format, Labels on Top) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training confusion matrix
cm_train = confusion_matrix(y_train, best_rf.predict(X_train_scaled))
cm_train_flip = np.array([[cm_train[1][1], cm_train[0][1]],
                           [cm_train[1][0], cm_train[0][0]]])

sns.heatmap(cm_train_flip, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Diabetes (1)', 'No Diabetes (0)'], 
            yticklabels=['Diabetes (1)', 'No Diabetes (0)'])
axes[0].set_title(f'Training Set\nAcc: {best_config["train_acc"]:.4f} | Recall: {best_config["train_recall"]:.4f}', 
                   fontweight='bold', pad=40)
axes[0].set_xlabel('')
axes[0].set_ylabel('Predicted')

# Move x-axis labels to top
axes[0].xaxis.set_ticks_position('top')
axes[0].xaxis.set_label_position('top')
axes[0].set_xlabel('Actual', fontsize=12)

# Validation confusion matrix
cm_val = confusion_matrix(y_val, y_val_pred_best)
cm_val_flip = np.array([[cm_val[1][1], cm_val[0][1]],
                         [cm_val[1][0], cm_val[0][0]]])

sns.heatmap(cm_val_flip, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['Diabetes (1)', 'No Diabetes (0)'], 
            yticklabels=['Diabetes (1)', 'No Diabetes (0)'])
axes[1].set_title(f'Validation Set\nAcc: {best_config["val_acc"]:.4f} | Recall: {best_config["val_recall"]:.4f}', 
                   fontweight='bold', pad=40)
axes[1].set_xlabel('')
axes[1].set_ylabel('Predicted')

# Move x-axis labels to top
axes[1].xaxis.set_ticks_position('top')
axes[1].xaxis.set_label_position('top')
axes[1].set_xlabel('Actual', fontsize=12)

plt.suptitle(f'Milestone 3: Confusion Matrices - Random Forest (n_estimators={int(best_config["n_trees"])})', 
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrices_m3.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [7d] Saved: confusion_matrices_m3.png")

# --- FIGURE 6: Feature Importance ---
fig, ax = plt.subplots(figsize=(10, 6))

importances = best_rf.feature_importances_
sorted_idx = np.argsort(importances)

colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_idx)))
ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], 
        color=colors, edgecolor='black')

for i, (val, name) in enumerate(zip(importances[sorted_idx], np.array(feature_names)[sorted_idx])):
    ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontweight='bold')

ax.set_xlabel('Feature Importance (Gini)', fontsize=12)
ax.set_title('Milestone 3: Random Forest Feature Importance\n(Supports clinical interpretability — Milestone 2 justification)', 
             fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('feature_importance_m3.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [7f] Saved: feature_importance_m3.png")

# ============================================================
# STEP 8: SUMMARY & NEXT STEPS
# ============================================================
print(f"\n\n{'=' * 65}")
print(f"  MILESTONE 3: TRAINING LOOP SUMMARY")
print(f"{'=' * 65}")

print(f"\n  MODEL: Random Forest")
print(f"  BEST CONFIGURATION: n_estimators = {int(best_config['n_trees'])}")

print(f"\n  TRAINING RESULTS:")
print(f"  {'Metric':<20} {'Training':>10} {'Validation':>12}")
print(f"  {'-'*44}")
print(f"  {'Accuracy':<20} {best_config['train_acc']:>10.4f} {best_config['val_acc']:>12.4f}")
print(f"  {'Recall ★':<20} {best_config['train_recall']:>10.4f} {best_config['val_recall']:>12.4f}")
print(f"  {'Precision':<20} {'---':>10} {best_config['val_precision']:>12.4f}")
print(f"  {'F1-Score':<20} {'---':>10} {best_config['val_f1']:>12.4f}")
print(f"  {'ROC AUC':<20} {'---':>10} {best_config['val_auc']:>12.4f}")
print(f"  {'Overfit Gap':<20} {best_config['overfit_gap']:>10.4f} {'⚠️ HIGH' if best_config['overfit_gap'] > 0.10 else '✅ OK':>12}")

print(f"\n  ADAP BENCHMARK (1988): 76% accuracy")
beat = "✅ YES" if best_config['val_acc'] >= 0.76 else "❌ NOT YET"
print(f"  Benchmark beaten?     {beat}")

# ============================================================
# STEP 9: KEY OBSERVATIONS & NEXT STEPS (Dynamic Analysis)
# ============================================================
print(f"\n  KEY OBSERVATIONS:")

# Observation 1: Overfitting analysis
overfit_gap = best_config['overfit_gap']
if overfit_gap > 0.20:
    print(f"    1. SEVERE OVERFITTING detected (gap: {overfit_gap:.4f})")
    print(f"       Training accuracy ({best_config['train_acc']:.4f}) is far higher than validation ({best_config['val_acc']:.4f})")
    print(f"       → Model has memorized training data instead of learning patterns")
elif overfit_gap > 0.10:
    print(f"    1. MODERATE OVERFITTING detected (gap: {overfit_gap:.4f})")
    print(f"       Training accuracy ({best_config['train_acc']:.4f}) is notably higher than validation ({best_config['val_acc']:.4f})")
    print(f"       → Model needs regularization to generalize better")
elif overfit_gap > 0.05:
    print(f"    1. MILD OVERFITTING detected (gap: {overfit_gap:.4f})")
    print(f"       Training ({best_config['train_acc']:.4f}) and validation ({best_config['val_acc']:.4f}) are somewhat close")
    print(f"       → Minor tuning may help close the gap further")
else:
    print(f"    1. GOOD FIT — minimal overfitting (gap: {overfit_gap:.4f})")
    print(f"       Training ({best_config['train_acc']:.4f}) and validation ({best_config['val_acc']:.4f}) are well aligned")
    print(f"       → Model generalizes well to unseen data")

# Observation 2: Recall analysis (PRIMARY METRIC)
val_recall = best_config['val_recall']
if val_recall < 0.50:
    print(f"    2. RECALL IS CRITICALLY LOW ({val_recall:.4f})")
    print(f"       More than half of diabetic patients are being MISSED")
    print(f"       → Must address class imbalance and lower classification threshold")
elif val_recall < 0.65:
    print(f"    2. RECALL IS BELOW TARGET ({val_recall:.4f})")
    print(f"       Too many diabetic patients are still being missed")
    print(f"       → Need class_weight='balanced' or SMOTE to boost minority class detection")
elif val_recall < 0.76:
    print(f"    2. RECALL IS MODERATE ({val_recall:.4f}) but below ADAP benchmark (0.76)")
    print(f"       → Hyperparameter tuning and threshold adjustment can improve this")
else:
    print(f"    2. RECALL IS STRONG ({val_recall:.4f}) — meets or exceeds ADAP benchmark (0.76)")
    print(f"       → Model is catching most diabetic patients correctly")

# Observation 3: Precision vs Recall tradeoff
val_prec = best_config['val_precision']
if val_prec > val_recall + 0.15:
    print(f"    3. PRECISION ({val_prec:.4f}) is much higher than RECALL ({val_recall:.4f})")
    print(f"       Model is too conservative — predicts diabetes only when very confident")
    print(f"       → Lower the classification threshold to catch more positive cases")
elif val_recall > val_prec + 0.15:
    print(f"    3. RECALL ({val_recall:.4f}) is much higher than PRECISION ({val_prec:.4f})")
    print(f"       Model is too aggressive — many false alarms")
    print(f"       → Raise the classification threshold to reduce false positives")
else:
    print(f"    3. PRECISION ({val_prec:.4f}) and RECALL ({val_recall:.4f}) are reasonably balanced")
    print(f"       → F1-Score ({best_config['val_f1']:.4f}) reflects this balance")

# Observation 4: ROC AUC analysis
val_auc = best_config['val_auc']
if val_auc >= 0.85:
    print(f"    4. ROC AUC is EXCELLENT ({val_auc:.4f})")
    print(f"       Model has strong ability to distinguish diabetic from non-diabetic")
elif val_auc >= 0.75:
    print(f"    4. ROC AUC is GOOD ({val_auc:.4f})")
    print(f"       Model has reasonable discriminative ability, but room for improvement")
elif val_auc >= 0.65:
    print(f"    4. ROC AUC is FAIR ({val_auc:.4f})")
    print(f"       Model struggles to separate classes — needs better features or tuning")
else:
    print(f"    4. ROC AUC is POOR ({val_auc:.4f})")
    print(f"       Model is barely better than random guessing — major changes needed")

# Observation 5: Feature importance
top_features = np.array(feature_names)[np.argsort(importances)[::-1]]
print(f"    5. Top 3 most important features: {top_features[0]}, {top_features[1]}, {top_features[2]}")
if top_features[0] == 'plas':
    print(f"       Glucose ('plas') is the #1 predictor — consistent with medical literature")
elif top_features[0] == 'mass':
    print(f"       BMI ('mass') is the #1 predictor — high BMI is a known diabetes risk factor")
elif top_features[0] == 'age':
    print(f"       Age is the #1 predictor — diabetes risk increases with age")
else:
    print(f"       {top_features[0]} is the #1 predictor — worth investigating clinical significance")

# Observation 6: Best n_estimators analysis
best_n = int(best_config['n_trees'])
if best_n <= 25:
    print(f"    6. Best performance at only {best_n} trees — model may benefit from more complex trees")
    print(f"       → Try deeper trees (increase max_depth) rather than more trees")
elif best_n >= 150:
    print(f"    6. Best performance at {best_n} trees — model needs many trees to perform")
    print(f"       → Individual trees are weak; try reducing max_features for diversity")
else:
    print(f"    6. Best performance at {best_n} trees — reasonable ensemble size")
    print(f"       → Focus optimization on tree structure (max_depth, min_samples_split)")

# ============================================================
# DYNAMIC NEXT STEPS (based on actual results)
# ============================================================
print(f"\n  NEXT STEPS (Milestone 4 - Model Optimization):")

step_num = 1

# Step: Address overfitting
if overfit_gap > 0.10:
    print(f"    {step_num}. REDUCE OVERFITTING (gap: {overfit_gap:.4f})")
    print(f"       → Limit tree depth (max_depth=5-10)")
    print(f"       → Increase min_samples_split (try 5, 10, 20)")
    print(f"       → Increase min_samples_leaf (try 2, 5, 10)")
    print(f"       → Reduce max_features (try 'sqrt' or 0.5)")
    step_num += 1

# Step: Improve recall
if val_recall < 0.76:
    print(f"    {step_num}. IMPROVE RECALL (currently {val_recall:.4f}, target ≥ 0.76)")
    if val_recall < 0.55:
        print(f"       → Apply SMOTE oversampling to balance classes")
        print(f"       → Use class_weight='balanced' in Random Forest")
        print(f"       → Lower classification threshold from 0.5 to 0.3-0.4")
    else:
        print(f"       → Try class_weight='balanced' in Random Forest")
        print(f"       → Fine-tune classification threshold (try 0.35-0.45)")
    step_num += 1

# Step: Precision-Recall balance
if abs(val_prec - val_recall) > 0.15:
    if val_prec > val_recall:
        print(f"    {step_num}. BALANCE PRECISION/RECALL (Prec: {val_prec:.4f} >> Rec: {val_recall:.4f})")
        print(f"       → Lower decision threshold to trade some precision for recall")
    else:
        print(f"    {step_num}. BALANCE PRECISION/RECALL (Rec: {val_recall:.4f} >> Prec: {val_prec:.4f})")
        print(f"       → Raise decision threshold to reduce false positives")
    step_num += 1

# Step: Hyperparameter tuning
print(f"    {step_num}. HYPERPARAMETER TUNING via GridSearchCV/RandomizedSearchCV")
print(f"       → n_estimators: [50, 100, 150, 200]")
if overfit_gap > 0.10:
    print(f"       → max_depth: [3, 5, 7, 10, None]  (currently unlimited → causing overfitting)")
else:
    print(f"       → max_depth: [5, 10, 15, None]")
print(f"       → min_samples_split: [2, 5, 10]")
print(f"       → min_samples_leaf: [1, 2, 5]")
step_num += 1

# Step: Imputation improvement
print(f"    {step_num}. CONSIDER IMPROVED IMPUTATION for insulin/skin columns")
print(f"       → Current median imputation created ~48% identical insulin values (125.0)")
print(f"       → Try KNN Imputation or Iterative Imputation for more realistic values")
step_num += 1

# Step: Accuracy benchmark
if best_config['val_acc'] < 0.76:
    print(f"    {step_num}. BEAT ADAP BENCHMARK (currently {best_config['val_acc']:.4f}, target ≥ 0.76)")
    print(f"       → Combined optimization above should push accuracy past 76%")
    step_num += 1

print(f"{'=' * 65}")

# ============================================================
# STEP 10: PRESENTATION SLIDES (Dynamic Terminal Output)
# ============================================================

def print_slide_training_progress(progress_df, best_n):
    print("\n" + "=" * 65)
    print("  TRAINING PROGRESS")
    print("=" * 65)
    print(f"  {'Trees':<10} {'Train Acc':<14} {'Val Acc':<14} {'Val Recall ★':<14}")
    print(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*14}")
    
    for _, row in progress_df.iterrows():
        n = int(row['n_trees'])
        marker = " ✅" if n == best_n else "   "
        print(f"  {n:<4}{marker}    "
              f"{row['train_acc']*100:>6.1f}%       "
              f"{row['val_acc']*100:>6.1f}%       "
              f"{row['val_recall']*100:>6.1f}%")
    
    best_row = progress_df[progress_df['n_trees'] == best_n].iloc[0]
    gap = best_row['overfit_gap']
    print()
    print(f"  ⚠️  Train = {best_row['train_acc']*100:.1f}% but Val = {best_row['val_acc']*100:.1f}%")
    print(f"  → Overfit Gap: {gap*100:.1f}%")
    print("=" * 65)


def print_slide_best_results(best_config, cm_val, val_class1_support):
    caught = cm_val[1][1]
    missed = cm_val[1][0]
    
    print("\n" + "=" * 65)
    print(f"  BEST MODEL RESULTS — {int(best_config['n_trees'])} Trees")
    print("=" * 65)
    print(f"""
  ┌──────────────────────┐    ┌──────────────────────────┐
  │  METRICS             │    │  CONFUSION MATRIX        │
  │                      │    │                          │
  │  Accuracy:  {best_config['val_acc']*100:>5.1f}%   │    │  Out of {val_class1_support} diabetic      │
  │  Recall ★:  {best_config['val_recall']*100:>5.1f}%   │    │  patients:               │
  │  Precision: {best_config['val_precision']*100:>5.1f}%   │    │                          │
  │  F1-Score:  {best_config['val_f1']*100:>5.1f}%   │    │  ✅ Caught: {caught:<3}            │
  │  ROC AUC:   {best_config['val_auc']*100:>5.1f}%   │    │  ❌ Missed: {missed:<3}            │
  │                      │    │                          │
  └──────────────────────┘    └──────────────────────────┘""")
    
    if best_config['val_acc'] >= 0.76:
        print(f"\n  ✅ ADAP Benchmark (76% accuracy) — BEATEN!")
    else:
        print(f"\n  ❌ ADAP Benchmark (76% accuracy) — NOT YET BEATEN")
    print("=" * 65)


def print_slide_crossval_features(cv_scores_stored, importances, feature_names):
    recall_scores = cv_scores_stored['recall']
    acc_scores = cv_scores_stored['accuracy']
    auc_scores = cv_scores_stored['roc_auc']
    
    top_3_idx = np.argsort(importances)[::-1][:3]
    top_3 = [(feature_names[i], importances[i]) for i in top_3_idx]
    
    print("\n" + "=" * 65)
    print("  CROSS-VALIDATION & TOP FEATURES")
    print("=" * 65)
    print(f"""
  CROSS-VALIDATION (5-Fold)       TOP FEATURES
                                  
  Avg Recall:   {recall_scores.mean()*100:>5.1f}%          #1  {top_3[0][0]} ({top_3[0][1]*100:.1f}%)
  Avg Accuracy: {acc_scores.mean()*100:>5.1f}%          #2  {top_3[1][0]} ({top_3[1][1]*100:.1f}%)
  Avg AUC:      {auc_scores.mean()*100:>5.1f}%          #3  {top_3[2][0]} ({top_3[2][1]*100:.1f}%)
  
  Recall ranged {recall_scores.min()*100:.0f}%–{recall_scores.max()*100:.0f}%
  → Model is unstable on minority class

  ─────────────────────────────────────────────────────────
  
  NEXT → Milestone 4: Optimization
  
  • Reduce overfitting (limit tree depth)
  • Improve recall (class balancing, threshold tuning)
  • Hyperparameter search (GridSearchCV)""")
    print("\n" + "=" * 65)


# --- PRINT ALL SLIDES ---
print("\n\n" + "=" * 65)
print("  MILESTONE 3: PRESENTATION SLIDE SUMMARIES")
print("=" * 65)

best_n = int(best_config['n_trees'])
val_class1_support = (y_val == 1).sum()
cm_val = confusion_matrix(y_val, y_val_pred_best)

print_slide_training_progress(progress_df, best_n)
print_slide_best_results(best_config, cm_val, val_class1_support)
print_slide_crossval_features(cv_scores_stored, importances, feature_names)