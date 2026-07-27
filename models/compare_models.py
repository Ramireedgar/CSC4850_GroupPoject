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
accuracies = []
precisions = []
recalls = []
f1_scores = []

if knn_data:
    models.append('KNN')
    accuracies.append(knn_data.get('test_accuracy', 0))
    precisions.append(knn_data.get('precision', 0))
    recalls.append(knn_data.get('recall', 0))
    f1_scores.append(knn_data.get('f1_score', 0))

if svm_data:
    models.append('SVM')
    accuracies.append(svm_data.get('test_accuracy', 0))
    precisions.append(svm_data.get('precision', 0))
    recalls.append(svm_data.get('recall', 0))
    f1_scores.append(svm_data.get('f1_score', 0))

if nn_data:
    models.append('Neural Network')
    accuracies.append(nn_data.get('test_accuracy', 0))
    precisions.append(nn_data.get('precision', 0))
    recalls.append(nn_data.get('recall', 0))
    f1_scores.append(nn_data.get('f1_score', 0))

print("Model Comparison:")
for i, model in enumerate(models):
    print(f"{model}: Accuracy = {accuracies[i]:.4f}, Precision = {precisions[i]:.4f}, Recall = {recalls[i]:.4f}, F1-Score = {f1_scores[i]:.4f}")

# Plotting the comparison as a grouped bar chart
x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='blue')
rects2 = ax.bar(x - 0.5*width, precisions, width, label='Precision', color='orange')
rects3 = ax.bar(x + 0.5*width, recalls, width, label='Recall', color='green')
rects4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='red')

ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison (Weighted Average)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.0)
ax.legend(loc='lower right')

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, 'model_comparison.png'))
plt.close()

print(f"Comparison plot saved to {os.path.join('results', 'model_comparison.png')}.")
