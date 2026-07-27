import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
import os

print("Loading PCA-reduced data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
preproc_dir = os.path.join(script_dir, '..', 'pre-processing')
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

X_train = pd.read_csv(os.path.join(preproc_dir, 'X_train_pca.csv'))
X_test  = pd.read_csv(os.path.join(preproc_dir, 'X_test_pca.csv'))
y_train = pd.read_csv(os.path.join(preproc_dir, 'y_train.csv')).values.ravel()
y_test  = pd.read_csv(os.path.join(preproc_dir, 'y_test.csv')).values.ravel()

# Check columns
print("Columns in PCA data:", X_train.columns.tolist()[:5])

print("Training Random Forest Classifier on PCA data (this may take a minute)...")
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
rf.fit(X_train, y_train)

print("Generating SHAP Explanations...")
# Use a very small subset of test data for SHAP to save computation time
X_test_sample = X_test.sample(n=100, random_state=42)

explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_test_sample)

# For a multi-class Random Forest, the shape is (n_samples, n_features, n_classes)
win_index = list(rf.classes_).index(1)
shap_values_win = shap_values[:, :, win_index]

print("Saving Global Summary Plot (Beeswarm)...")
plt.figure()
shap.summary_plot(shap_values_win, show=False)
plt.title('SHAP Summary on PCA Data')
plt.savefig(os.path.join(results_dir, 'shap_pca_summary.png'), bbox_inches='tight')
plt.close()

print(f"Done! Plot saved as shap_pca_summary.png in {results_dir}")
