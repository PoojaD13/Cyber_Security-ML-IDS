import matplotlib.pyplot as plt
import numpy as np

# ========= Baseline Models =========
models = [
    "Logistic Regression", 
    "Decision Tree", 
    "Random Forest", 
    "SVM", 
    "KNN"
]

# ========= Your Provided Metrics =========
accuracy  = [0.621406, 1.000000 , 0.947662, 0.725341 , 0.739627]
precision = [0.943371, 1.000000, 0.999991, 0.990761, 0.986466]
recall    = [0.472093, 1.000000, 0.923111, 0.602073, 0.626038]
f1_score  = [0.629276, 1.000000, 0.960014, 0.748993 , 0.765971]

# ========= Plotting =========
x = np.arange(len(models))
width = 0.2  # bar width

fig, ax = plt.subplots(figsize=(10,6))
bar1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy')
bar2 = ax.bar(x - 0.5*width, precision, width, label='Precision')
bar3 = ax.bar(x + 0.5*width, recall, width, label='Recall')
bar4 = ax.bar(x + 1.5*width, f1_score, width, label='F1 Score')

# Formatting the plot
ax.set_xlabel("Models")
ax.set_ylabel("Scores")
ax.set_title("Baseline Model Performance on UNSW-NB15")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right")
ax.set_ylim(0.0, 1.1)
ax.legend(loc='lower right')

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bar1)
add_labels(bar2)
add_labels(bar3)
add_labels(bar4)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# ==============Decision tree Purned=================
import matplotlib.pyplot as plt
import numpy as np

# Models to compare
models = ["Decision Tree (Baseline)", "Decision Tree (Pruned)"]

# Metrics from your experiment
accuracy  = [1.000000, 0.591903776070628]
precision = [1.000000, 0.8048562642746865]
recall    = [1.000000, 0.5285610142365156]
f1_score  = [1.000000, 0.6380832920278786]

# Plotting
x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(8,5))
b1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy')
b2 = ax.bar(x - 0.5*width, precision, width, label='Precision')
b3 = ax.bar(x + 0.5*width, recall, width, label='Recall')
b4 = ax.bar(x + 1.5*width, f1_score, width, label='F1 Score')

# Labels and formatting
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("Decision Tree: Baseline vs Pruned")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylim(0.0, 1.1)
ax.legend(loc='lower right')

# Add value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(b1)
add_labels(b2)
add_labels(b3)
add_labels(b4)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ===================Random Forest Alg tuned ================

import matplotlib.pyplot as plt
import numpy as np

# Labels
models = ["Random Forest (Baseline)", "Random Forest (Tuned)"]

# Final metrics (from your message)
accuracy  = [0.949521, 0.947434]
precision = [0.999991, 0.999964]
recall    = [0.925843, 0.922801]
f1_score  = [0.961489, 0.959834]

# X locations
x = np.arange(len(models))
width = 0.2

# Plot setup
fig, ax = plt.subplots(figsize=(9,5))
b1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy')
b2 = ax.bar(x - 0.5*width, precision, width, label='Precision')
b3 = ax.bar(x + 0.5*width, recall, width, label='Recall')
b4 = ax.bar(x + 1.5*width, f1_score, width, label='F1 Score')

# Axis formatting
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("Random Forest: Baseline vs Tuned")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylim(0.90, 1.01)
ax.legend(loc='lower right')

# Annotate bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(b1)
add_labels(b2)
add_labels(b3)
add_labels(b4)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
