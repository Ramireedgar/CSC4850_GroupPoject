import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.neural_network import MLPClassifier
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

print("Training Neural Network (MLPClassifier) on PCA data...")
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True)
nn.fit(X_train, y_train)

win_index = list(nn.classes_).index(1)

def predict_win_prob(x):
    return nn.predict_proba(x)[:, win_index]

print("Generating SHAP Explanations using KernelExplainer (this may take a bit longer)...")
# For KernelExplainer, we summarize the background data with k-means to speed it up
background = shap.kmeans(X_train, 10)
explainer = shap.KernelExplainer(predict_win_prob, background)

# Use a small subset of test data for SHAP to save computation time
X_test_sample = X_test.sample(n=30, random_state=42)

# Get SHAP values
shap_values_win = explainer.shap_values(X_test_sample)

print(f"shap_values_win shape: {np.array(shap_values_win).shape}")
print(f"X_test_sample shape: {X_test_sample.shape}")

print("Saving Global Summary Plot (Beeswarm)...")
plt.figure()
shap.summary_plot(np.array(shap_values_win), X_test_sample, show=False)
plt.title('SHAP Summary (Neural Network on PCA Data)')
plt.savefig(os.path.join(results_dir, 'shap_nn_pca_summary.png'), bbox_inches='tight')
plt.close()

print(f"Done! Plot saved as shap_nn_pca_summary.png in {results_dir}")
