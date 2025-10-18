# regularized dt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import time

# Train Decision Tree with regularization
dt = DecisionTreeClassifier(
    random_state=42,
    max_depth=8,             # limit tree depth
    min_samples_split=50,    # at least 50 samples to split
    min_samples_leaf=25,     # each leaf must have at least 25 samples
    max_features="sqrt",     # use only subset of features
    class_weight="balanced", # handle imbalance
    ccp_alpha=0.001          # pruning to reduce complexity
)

start = time.time()
dt.fit(X_train, y_train)
end = time.time()

# Predictions
y_pred = dt.predict(X_test)
y_proba = dt.predict_proba(X_test)[:, 1]

# Metrics
print("Decision Tree Performance (Regularized):")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Training Time: ", round(end - start, 2), "seconds")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label="Decision Tree (AUC = %.2f)" % roc_auc_score(y_test, y_proba))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree")
plt.legend()
plt.show()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
plt.plot(rec, prec, label="Decision Tree")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Decision Tree")
plt.legend()
plt.show()