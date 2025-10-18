#comparision of baseline and tuned rf

import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# ========== Baseline Random Forest ==========
start = time.time()
baseline_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
baseline_rf.fit(X_train, y_train)
y_pred_baseline = baseline_rf.predict(X_test)
y_proba_baseline = baseline_rf.predict_proba(X_test)[:,1]
baseline_time = time.time() - start

# Baseline metrics
baseline_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_baseline),
    "Precision": precision_score(y_test, y_pred_baseline),
    "Recall": recall_score(y_test, y_pred_baseline),
    "F1 Score": f1_score(y_test, y_pred_baseline),
    "AUC": roc_auc_score(y_test, y_proba_baseline),
    "Training Time": baseline_time
}

# ========== Tuned Random Forest ==========
start = time.time()
tuned_rf = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
tuned_rf.fit(X_train, y_train)
y_pred_tuned = tuned_rf.predict(X_test)
y_proba_tuned = tuned_rf.predict_proba(X_test)[:,1]
tuned_time = time.time() - start

# Tuned metrics
tuned_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_tuned),
    "Precision": precision_score(y_test, y_pred_tuned),
    "Recall": recall_score(y_test, y_pred_tuned),
    "F1 Score": f1_score(y_test, y_pred_tuned),
    "AUC": roc_auc_score(y_test, y_proba_tuned),
    "Training Time": tuned_time
}

# ========== Compare Results ==========
results_df = pd.DataFrame([baseline_metrics, tuned_metrics], index=["Baseline RF", "Tuned RF"])
print("\n===== Random Forest Comparison =====")
print(results_df)

# ========== Plot ROC Curve ==========
fpr_b, tpr_b, _ = roc_curve(y_test, y_proba_baseline)
fpr_t, tpr_t, _ = roc_curve(y_test, y_proba_tuned)

plt.figure(figsize=(10,5))
plt.plot(fpr_b, tpr_b, label=f'Baseline RF (AUC = {baseline_metrics["AUC"]:.4f})')
plt.plot(fpr_t, tpr_t, label=f'Tuned RF (AUC = {tuned_metrics["AUC"]:.4f})', linestyle="--")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Baseline vs Tuned Random Forest")
plt.legend()
plt.show()

# ========== Plot Precision-Recall Curve ==========
prec_b, rec_b, _ = precision_recall_curve(y_test, y_proba_baseline)
prec_t, rec_t, _ = precision_recall_curve(y_test, y_proba_tuned)

plt.figure(figsize=(10,5))
plt.plot(rec_b, prec_b, label="Baseline RF")
plt.plot(rec_t, prec_t, label="Tuned RF", linestyle="--")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Baseline vs Tuned Random Forest")
plt.legend()
plt.show()