import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import json
import os

print("Loading un-reduced scaled data...")
# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
preproc_dir = os.path.join(script_dir, '..', 'pre-processing')
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

X_train = pd.read_csv(os.path.join(preproc_dir, 'X_train_scaled.csv'))
X_test  = pd.read_csv(os.path.join(preproc_dir, 'X_test_scaled.csv'))
y_train = pd.read_csv(os.path.join(preproc_dir, 'y_train.csv')).values.ravel()
y_test  = pd.read_csv(os.path.join(preproc_dir, 'y_test.csv')).values.ravel()

print(f"Training features shape: {X_train.shape}")
print("Training Neural Network (MLPClassifier) (this may take a minute)...")
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True)
nn.fit(X_train, y_train)

y_pred = nn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Neural Network Test Accuracy: {acc:.4f}")

result = {
    "model": "Neural Network (XAI Base)",
    "test_accuracy": round(acc, 4)
}
with open(os.path.join(script_dir, 'nn_xai_result.json'), 'w') as f:
    json.dump(result, f, indent=2)

print("\nGenerating SHAP Explanations using KernelExplainer...")
# We use a small subset of test data for SHAP to save computation time
X_test_sample = X_test.sample(n=100, random_state=42)

win_index = list(nn.classes_).index(1)

def predict_win_prob(x):
    return nn.predict_proba(x)[:, win_index]

# Summarize the background data with k-means to speed it up
background = shap.kmeans(X_train, 10)
explainer = shap.KernelExplainer(predict_win_prob, background)

# Get SHAP values
print("Calculating SHAP values (this may take a bit longer)...")
shap_values_array = explainer.shap_values(X_test_sample)

# 1. Global Summary Plot
print("Saving Global Summary Plot (Beeswarm)...")
plt.figure()
shap.summary_plot(np.array(shap_values_array), X_test_sample, show=False)
plt.savefig(os.path.join(results_dir, 'shap_nn_summary_plot.png'), bbox_inches='tight')
plt.close()

# 2. Local Waterfall Plot for a single match (First instance in our sample)
print("Saving Local Match Explanation (Waterfall Plot)...")

# Construct an Explanation object for the waterfall plot
expected_val = explainer.expected_value
if isinstance(expected_val, (np.ndarray, list)):
    expected_val = expected_val[0]

explanation = shap.Explanation(
    values=shap_values_array[0],
    base_values=expected_val,
    data=X_test_sample.iloc[0].values,
    feature_names=X_test_sample.columns.tolist()
)

plt.figure(figsize=(10, 6))
shap.waterfall_plot(explanation, show=False)
plt.savefig(os.path.join(results_dir, 'shap_nn_local_match.png'), bbox_inches='tight')
plt.close()

print("All XAI tasks completed successfully!")
