import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import os

print("Loading un-reduced scaled data...")
# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
preproc_dir = os.path.join(script_dir, '..', 'pre-processing')

X_train = pd.read_csv(os.path.join(preproc_dir, 'X_train_scaled.csv'))
X_test  = pd.read_csv(os.path.join(preproc_dir, 'X_test_scaled.csv'))
y_train = pd.read_csv(os.path.join(preproc_dir, 'y_train.csv')).values.ravel()
y_test  = pd.read_csv(os.path.join(preproc_dir, 'y_test.csv')).values.ravel()

print(f"Training features shape: {X_train.shape}")
print("Training Random Forest Classifier (this may take a minute)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Test Accuracy: {acc:.4f}")

result = {
    "model": "Random Forest (XAI Base)",
    "test_accuracy": round(acc, 4)
}
with open(os.path.join(script_dir, 'rf_xai_result.json'), 'w') as f:
    json.dump(result, f, indent=2)

print("\nGenerating SHAP Explanations...")
# We use a subset of test data for SHAP to save computation time
X_test_sample = X_test.sample(n=300, random_state=42)

explainer = shap.TreeExplainer(rf)
# The modern SHAP API returns an Explanation object directly
shap_values = explainer(X_test_sample)

# For a multi-class Random Forest, the shape is (n_samples, n_features, n_classes)
# We will focus on predicting a "Win" (class 1, which might be at index 1 depending on rf.classes_)
win_index = list(rf.classes_).index(1)
shap_values_win = shap_values[:, :, win_index]

# 1. Global Summary Plot
print("Saving Global Summary Plot (Beeswarm)...")
plt.figure()
shap.summary_plot(shap_values_win, show=False)
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, 'shap_summary_plot.png'), bbox_inches='tight')
plt.close()

# 2. Local Waterfall Plot for a single match (First instance in our sample)
print("Saving Local Match Explanation (Waterfall Plot)...")
plt.figure(figsize=(10, 6))
shap.waterfall_plot(shap_values_win[0], show=False)
plt.savefig(os.path.join(results_dir, 'shap_local_match.png'), bbox_inches='tight')
plt.close()

print("All XAI tasks completed successfully!")
