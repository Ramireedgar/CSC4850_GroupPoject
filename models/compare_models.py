import json
import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

def load_json(filename):
    filepath = os.path.join(script_dir, filename)
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

knn_data = load_json('knn_best_result.json')
svm_data = load_json('svm_best_result.json')
nn_data = load_json('nn_best_result.json')

models = []
test_accuracies = []

if knn_data:
    models.append('KNN')
    test_accuracies.append(knn_data.get('test_accuracy', 0))

if svm_data:
    models.append('SVM')
    test_accuracies.append(svm_data.get('test_accuracy', 0))

if nn_data:
    models.append('Neural Network')
    test_accuracies.append(nn_data.get('test_accuracy', 0))

print("Model Comparison:")
for model, acc in zip(models, test_accuracies):
    print(f"{model}: Test Accuracy = {acc:.4f}")

# Plotting the comparison
plt.figure(figsize=(8, 5))
bars = plt.bar(models, test_accuracies, color=['blue', 'green', 'orange'])
plt.ylabel('Test Accuracy')
plt.title('Model Test Accuracy Comparison')
plt.ylim(0, 1.0)

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')

results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, 'model_comparison.png'))
plt.close()

print(f"Comparison plot saved to {os.path.join('results', 'model_comparison.png')}.")
