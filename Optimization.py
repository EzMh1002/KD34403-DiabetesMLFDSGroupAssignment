import pandas as pd
import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (recall_score, precision_score, f1_score, roc_auc_score, accuracy_score)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve

print("Milestone 4 - Optimization: RF + Sampling + Tuning")

#Load data
df = pd.read_csv('cleaned_diabetes_data.csv')
X = df.drop('target', axis=1)
y = df['target']

np.random.seed(67)

def split(indices, train_ratio=0.6, val_ratio=0.2):
    n = len(indices)
    return indices[:int(n*train_ratio)], indices[int(n*train_ratio):int(n*(train_ratio+val_ratio))], indices[int(n*(train_ratio+val_ratio)):]

idx0 = y[y==0].index.tolist()
idx1 = y[y==1].index.tolist()

np.random.shuffle(idx0)
np.random.shuffle(idx1)

tr0, va0, te0 = split(idx0)
tr1, va1, te1 = split(idx1)

train_idx = tr0 + tr1
val_idx = va0 + va1
test_idx = te0 + te1

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_val, y_val = X.loc[val_idx], y.loc[val_idx]
X_test, y_test = X.loc[test_idx], y.loc[test_idx]

#Parameter
parameter = {'n_estimators': [100, 150],'max_depth': [5, 7, 10],'min_samples_split': [5, 10],'min_samples_leaf': [2, 5],'max_features': ['sqrt', 0.5]}

keys = list(parameter.keys())
combinations = list(product(*parameter.values()))

def evaluate_model(X_tr, y_tr, label, sampler=None, use_balanced_rf=False):

    print(f"\nSampling method: {label}")
    results = []
    val_recall_progress = []
    
    for combo in combinations:
        params = dict(zip(keys, combo))

        # Apply Sampling
        X_tr_local, y_tr_local = X_tr.copy(), y_tr.copy()
        if sampler is not None:
            X_tr_local, y_tr_local = sampler.fit_resample(X_tr_local, y_tr_local)

        # Scale after sampling
        scaler = StandardScaler()
        X_tr_local = scaler.fit_transform(X_tr_local)
        X_val_scaled = scaler.transform(X_val)

        if use_balanced_rf:
            model = BalancedRandomForestClassifier(**params, random_state=67, n_jobs=-1)
        else:
            model = RandomForestClassifier(**params, class_weight={0:1,1:3}, random_state=67, n_jobs=-1)

        model.fit(X_tr_local, y_tr_local)

        train_recall = recall_score(y_tr_local, model.predict(X_tr_local))
        y_val_pred = model.predict(X_val_scaled)

        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        
        val_recall_progress.append(val_recall)
        
        gap = train_recall - val_recall
        score = val_f1 - 0.5 * gap

        results.append((params, val_recall, val_f1, gap, score))

    best = max(results, key=lambda x: x[4])
    best_params = best[0]

    print("Best Params:", best_params)

    # Final model
    X_tr_local, y_tr_local = X_tr.copy(), y_tr.copy()
    if sampler is not None:
        X_tr_local, y_tr_local = sampler.fit_resample(X_tr_local, y_tr_local)

    scaler = StandardScaler()
    X_tr_local = scaler.fit_transform(X_tr_local)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if use_balanced_rf:
        model = BalancedRandomForestClassifier(**best_params, random_state=67, n_jobs=-1)
    else:
        model = RandomForestClassifier(**best_params, class_weight={0:1,1:3}, random_state=67, n_jobs=-1)

    model.fit(X_tr_local, y_tr_local)

    # Threshold
    probs = model.predict_proba(X_val_scaled)[:,1]
    best_t, best_f1 = 0.5, 0

    for t in np.arange(0.3, 0.7, 0.05):
        f1 = f1_score(y_val, (probs > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print("Best Threshold:", best_t)

    # Test
    test_probs = model.predict_proba(X_test_scaled)[:,1]
    test_pred = (test_probs > best_t).astype(int)

    acc = accuracy_score(y_test, test_pred)
    rec = recall_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    auc = roc_auc_score(y_test, test_probs)

    print(f"Acc:{acc:.3f} Rec:{rec:.3f} Prec:{prec:.3f} F1:{f1:.3f} AUC:{auc:.3f}")

    return best_params, best_t, model, X_tr_local, X_test_scaled, test_pred, probs, val_recall_progress


#K-Fold Parameter Search
def kfold_param_search(X, y, label, sampler=None, use_balanced_rf=False):

    print(f"\nK-FOLD PARAM SEARCH: {label}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=67)
    results = []

    for combo in combinations:
        params = dict(zip(keys, combo))
        scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            if sampler is not None:
                X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val_fold = scaler.transform(X_val_fold)

            if use_balanced_rf:
                model = BalancedRandomForestClassifier(**params, random_state=67, n_jobs=-1)
            else:
                model = RandomForestClassifier(**params, class_weight={0:1,1:3}, random_state=67, n_jobs=-1)

            model.fit(X_tr, y_tr)
            scores.append(f1_score(y_val_fold, model.predict(X_val_fold)))

        results.append((params, np.mean(scores)))

    best = max(results, key=lambda x: x[1])
    print("Best Params:", best[0])

    return best[0]


#K-Fold Evaluation

def evaluate_kfold_sampling(X, y, label, params, best_t, sampler=None, use_balanced_rf=False):

    print(f"\nK-FOLD: {label}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=67)

    acc, rec, prec, f1, auc = [], [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        if sampler is not None:
            X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val_fold = scaler.transform(X_val_fold)

        if use_balanced_rf:
            model = BalancedRandomForestClassifier(**params, random_state=67, n_jobs=-1)
        else:
            model = RandomForestClassifier(**params, class_weight={0:1,1:3}, random_state=67, n_jobs=-1)

        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_val_fold)[:,1]
        pred = (proba > best_t).astype(int)

        acc.append(accuracy_score(y_val_fold, pred))
        rec.append(recall_score(y_val_fold, pred))
        prec.append(precision_score(y_val_fold, pred))
        f1.append(f1_score(y_val_fold, pred))
        auc.append(roc_auc_score(y_val_fold, proba))

    print(f"Acc:{np.mean(acc):.3f} Rec:{np.mean(rec):.3f} Prec:{np.mean(prec):.3f} F1:{np.mean(f1):.3f}")

    return (label, np.mean(acc), np.mean(rec), np.mean(prec), np.mean(f1), np.mean(auc))



# Method run

methods = [
    ("No Sampling", None, False),
    ("SMOTE", SMOTE(random_state=67), False),
    ("Borderline-SMOTE", BorderlineSMOTE(random_state=67), False),
    ("SMOTE-Tomek", SMOTETomek(random_state=67), False),
    ("SMOTE-ENN", SMOTEENN(random_state=67), False),
    ("ADASYN", ADASYN(random_state=67), False),
    ("UnderSampling", RandomUnderSampler(random_state=67), False),
    ("Balanced RF", None, True)
]
visualization_store = {}
results_all = []


print("\nMANUAL SPLIT RESULTS")


best_params_store = {}

for label, sampler, use_balanced_rf in methods:

    print("\n" + "-"*40)

    params, threshold, model, X_train_scaled, X_test_scaled, test_pred, probs, val_recall_progress= evaluate_model(
        X_train, y_train,
        label,
        sampler,
        use_balanced_rf
    )

    # re-evaluate on test for storing
    X_tr_local, y_tr_local = X_train.copy(), y_train.copy()
    if sampler is not None:
        X_tr_local, y_tr_local = sampler.fit_resample(X_tr_local, y_tr_local)

    scaler = StandardScaler()
    X_tr_local = scaler.fit_transform(X_tr_local)
    X_test_scaled = scaler.transform(X_test)

    if use_balanced_rf:
        model = BalancedRandomForestClassifier(**params, random_state=67, n_jobs=-1)
    else:
        model = RandomForestClassifier(**params, class_weight={0:1,1:3}, random_state=67, n_jobs=-1)

    model.fit(X_tr_local, y_tr_local)

    proba = model.predict_proba(X_test_scaled)[:,1]
    pred = (proba > threshold).astype(int)

    acc = accuracy_score(y_test, pred)
    rec = recall_score(y_test, pred)
    prec = precision_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)

    results_all.append((label, acc, rec, prec, f1, auc))
    visualization_store[label] = {
    'model': model,
    'params': params,
    'threshold': threshold,
    'X_train_scaled': X_tr_local,
    'y_train': y_tr_local,
    'X_test_scaled': X_test_scaled,
    'test_pred': pred,
    'test_proba': proba,
    'val_recall_progress': val_recall_progress,
    'sampler': sampler,
    'use_balanced_rf': use_balanced_rf
}
kfold_results = []

for label, sampler, use_balanced_rf in methods:

    print("\n" + "="*50)

    params, threshold, model, X_train_scaled, X_test_scaled, test_pred, probs, val_f1_progress = evaluate_model(X_train, y_train, label, sampler, use_balanced_rf)
    best_params = kfold_param_search(X, y, label, sampler, use_balanced_rf)

    best_params_store[label] = best_params 
    
    res = evaluate_kfold_sampling(X, y, label, best_params, threshold, sampler, use_balanced_rf)
    kfold_results.append(res)

best_manual = max(results_all, key=lambda x: x[4])   # F1
best_kfold = max(kfold_results, key=lambda x: x[4])

if best_kfold[4] >= best_manual[4]:
    final = best_kfold
    source = "K-Fold"
else:
    final = best_manual
    source = "Manual Split"
    
best_label = final[0]
best_metrics = final


print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)

print("\nManual Split:")
for r in results_all:
    print(f"{r[0]:<18} | Acc:{r[1]:.3f} | Rec:{r[2]:.3f} | Prec:{r[3]:.3f} | F1:{r[4]:.3f} | AUC:{r[5]:.3f}")

print("\nK-Fold:")
for r in kfold_results:
    print(f"{r[0]:<18} | Acc:{r[1]:.3f} | Rec:{r[2]:.3f} | Prec:{r[3]:.3f} | F1:{r[4]:.3f} | AUC:{r[5]:.3f}")
    
print("FINAL SELECTED MODEL\n")

print(f"Selected from: {source}")
print(f"Model: {final[0]}")
print(f"Accuracy : {final[1]:.3f}")
print(f"Recall   : {final[2]:.3f}")
print(f"Precision: {final[3]:.3f}")
print(f"F1-Score : {final[4]:.3f}")
print(f"AUC      : {final[5]:.3f}")

print("\nBest Parameters Used:")

if best_label in best_params_store:
    for k, v in best_params_store[best_label].items():
        print(f"{k}: {v}")
else:
    print("No parameters stored (manual split model)")
    
# Visualisation

print("\nGenerating Visualizations...")

best_data = visualization_store[best_label]

model = best_data['model']
params = best_data['params']
threshold = best_data['threshold']
X_train_scaled = best_data['X_train_scaled']
y_train_local = best_data['y_train']
X_test_scaled = best_data['X_test_scaled']
test_pred = best_data['test_pred']
test_proba = best_data['test_proba']
val_recall_progress = best_data['val_recall_progress']

feature_names = X.columns.tolist()

#Validation Recall Progression

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(range(1, len(val_recall_progress)+1),
        val_recall_progress,
        marker='o',
        linewidth=2)

ax.set_title(f'Validation Recall Progression - {best_label}',
             fontweight='bold')

ax.set_xlabel('Parameter Combination')
ax.set_ylabel('Validation ')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimization_f1_progress.png', dpi=150)
plt.close()

print("Saved: optimization_f1_progress.png")

#Manual Split Comparison

results_df = pd.DataFrame(results_all,
                          columns=['Method','Accuracy','Recall',
                                   'Precision','F1','AUC'])

fig, ax = plt.subplots(figsize=(12,6))

x = np.arange(len(results_df))

ax.plot(x, results_df['Accuracy'], marker='o', label='Accuracy')
ax.plot(x,results_df['Recall'],marker='s',linewidth=3,label='Recall')
ax.plot(x, results_df['Precision'], marker='^', label='Precision')
ax.plot(x, results_df['F1'], marker='D', label='F1')
ax.plot(x, results_df['AUC'], marker='v', label='AUC')

ax.set_xticks(x)
ax.set_xticklabels(results_df['Method'], rotation=20)

ax.set_ylim(0.4, 1.0)

ax.set_title('Sampling Method Performance Comparison',
             fontweight='bold')

ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sampling_method_comparison.png', dpi=150)
plt.close()

print("Saved: sampling_method_comparison.png")

#Learning Curve

fig, ax = plt.subplots(figsize=(10,6))

train_sizes, train_scores, val_scores = learning_curve(model,X_train_scaled,y_train_local,train_sizes=np.linspace(0.1, 1.0, 10),cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=67),scoring='recall_macro',error_score=0)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

ax.plot(train_sizes,train_mean,marker='o', linewidth=2,label='Training Recall')

ax.plot(train_sizes, val_mean,marker='s', linewidth=2,label='Cross-Val Recall')

ax.set_title(f'Learning Curve - {best_label}', fontweight='bold')

ax.set_xlabel('Training Samples')
ax.set_ylabel('Recall')

ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curve_optimization.png', dpi=150)
plt.close()

print("Saved: learning_curve_optimization.png")

#Confusion Matrix

fig, ax = plt.subplots(figsize=(6,5))

cm = confusion_matrix(test_pred, y_test)

sns.heatmap(cm,annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes','Diabetes'], yticklabels=['No Diabetes','Diabetes'])

ax.set_title(f'Confusion Matrix - {best_label}', fontweight='bold')

ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrix_optimization.png', dpi=150)
plt.close()

print("Saved: confusion_matrix_optimization.png")

#Feature Importance

if hasattr(model, 'feature_importances_'):

    fig, ax = plt.subplots(figsize=(10,6))

    importances = model.feature_importances_

    sorted_idx = np.argsort(importances)

    ax.barh(np.array(feature_names)[sorted_idx],
            importances[sorted_idx])

    ax.set_title(f'Feature Importance - {best_label}',
                 fontweight='bold')

    ax.set_xlabel('Importance')

    plt.tight_layout()
    plt.savefig('feature_importance_optimization.png',
                dpi=150)

    plt.close()

    print("Saved: feature_importance_optimization.png")