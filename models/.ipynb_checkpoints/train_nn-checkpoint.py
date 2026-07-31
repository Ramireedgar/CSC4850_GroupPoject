import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
preproc_dir = os.path.join(script_dir, '..', 'pre-processing')

# Load PCA-reduced data from the pre-processing folder
X_train = pd.read_csv(os.path.join(preproc_dir, 'X_train_pca.csv'))
X_test  = pd.read_csv(os.path.join(preproc_dir, 'X_test_pca.csv'))
y_train = pd.read_csv(os.path.join(preproc_dir, 'y_train.csv')).values.ravel()
y_test  = pd.read_csv(os.path.join(preproc_dir, 'y_test.csv')).values.ravel()

print('Training Features Shape:', X_train.shape)
print('Testing Features Shape: ', X_test.shape)

# Define and train the Neural Network (MLPClassifier)
# Using a simple architecture: 1 hidden layer with 100 neurons, max_iter=500 to ensure convergence
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(nn, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
cv_accuracy = scores.mean()
print(f'Cross-Validation Accuracy: {cv_accuracy:.4f}')

# Train on the full training set
nn.fit(X_train, y_train)

# Predict on test set
y_pred = nn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Save training loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(nn.loss_curve_, color='steelblue', linewidth=2, label='Training Loss')
if hasattr(nn, 'validation_scores_'):
    ax2 = ax.twinx()
    ax2.plot(nn.validation_scores_, color='darkorange', linewidth=2, linestyle='--', label='Validation Accuracy')
    ax2.set_ylabel('Validation Accuracy', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
ax.set_xlabel('Iteration (Epoch)', fontsize=11)
ax.set_ylabel('Training Loss', fontsize=11)
ax.set_title('Neural Network Training Loss Curve', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'nn_loss_curve.png'), dpi=150)
plt.close()
print("Saved: nn_loss_curve.png")

# Save confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Neural Network Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, 'nn_confusion_matrix.png'))
plt.close()

# Calculate additional metrics
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Save results to json
result = {
    "model": "Neural Network (MLP)",
    "best_params": {
        "hidden_layer_sizes": [100],
        "max_iter": 500,
        "early_stopping": True
    },
    "cv_accuracy": round(cv_accuracy, 4),
    "test_accuracy": round(test_accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4)
}

with open(os.path.join(script_dir, 'nn_best_result.json'), 'w') as f:
    json.dump(result, f, indent=2)

print("Finished training Neural Network. Results saved to nn_best_result.json.")
