# ========Finallized code =========================
# Network Traffic Classification
# Dataset: UNSW-NB15 (Train/Test split)
# Models: LogisticRegression, DecisionTree, RandomForest, SVM, KNN
# =================================

# !pip -q install scikit-learn pandas matplotlib

import time, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ======================
# 1. Load Train & Test
# ======================
train = pd.read_csv("path of the dataset")
test  = pd.read_csv("path of the dataset")

# Target column
target_col = "label"

X_train = train.drop(columns=[target_col])
y_train = train[target_col]
X_test  = test.drop(columns=[target_col])
y_test  = test[target_col]

cat_cols = X_train.select_dtypes(include=['object']).columns

for col in cat_cols:
    # Combine train and test to learn all categories
    le = LabelEncoder()
    le.fit(pd.concat([X_train[col], X_test[col]], axis=0).astype(str))

    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col]  = le.transform(X_test[col].astype(str))


# Standardize features for models sensitive to scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ======================
# 2. Define Models
# ======================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# ======================
# 3. Helpers
# ======================
def plot_confusion(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title, ylabel='Actual', xlabel='Predicted')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.tight_layout()
    plt.show()

def plot_roc_pr(y_true, y_proba, title):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} - ROC Curve")
    plt.legend()
    plt.show()
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.plot(recall, precision, label=f"AP = {ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} - Precision-Recall Curve")
    plt.legend()
    plt.show()

def plot_feature_importance_rf(model, feature_names, top=20):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top]
    plt.bar(range(len(idx)), importances[idx])
    plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=90)
    plt.title("Top Feature Importances (RandomForest)")
    plt.show()

# ======================
# 4. Train & Evaluate
# ======================
results = []
classes = ["Benign (0)", "Attack (1)"]

for name, model in models.items():
    print(f"\n▶ Training {name} ...")
    t0 = time.time()

    # Scale-sensitive models
    if name in ["Logistic Regression", "SVM", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred  = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

    train_time = time.time() - t0

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary")
    rec  = recall_score(y_test, y_pred, average="binary")
    f1   = f1_score(y_test, y_pred, average="binary")

    results.append([name, acc, prec, rec, f1, train_time])

    # Confusion Matrix
    plot_confusion(y_test, y_pred, classes, f"{name} - Confusion Matrix")

    # ROC + PR
    plot_roc_pr(y_test, y_proba, name)

    # Feature Importance for RF
    if name == "Random Forest":
        plot_feature_importance_rf(model, X_train.columns)

# ======================
# 5. Summary Table
# ======================
results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1","TrainTime"])
print("\n=== Summary Results ===")
print(results_df)