import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

# Load PCA-reduced data from the pre-processing folder
X_train = pd.read_csv('../pre-processing/X_train_pca.csv')
X_test  = pd.read_csv('../pre-processing/X_test_pca.csv')
y_train = pd.read_csv('../pre-processing/y_train.csv').values.ravel()
y_test  = pd.read_csv('../pre-processing/y_test.csv').values.ravel()

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

# Save confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Neural Network Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('nn_confusion_matrix.png')
plt.close()

# Save results to json
result = {
    "model": "Neural Network (MLP)",
    "best_params": {
        "hidden_layer_sizes": [100],
        "max_iter": 500,
        "early_stopping": True
    },
    "cv_accuracy": round(cv_accuracy, 4),
    "test_accuracy": round(test_accuracy, 4)
}

with open('nn_best_result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("Finished training Neural Network. Results saved to nn_best_result.json.")
