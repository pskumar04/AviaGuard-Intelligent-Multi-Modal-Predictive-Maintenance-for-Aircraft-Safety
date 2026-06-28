import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# Sample data (replace with your actual model probabilities)
np.random.seed(42)
n_samples = 500
n_classes = 16

# Generate sample data (replace with your actual y_test and y_score)
y_test = np.random.randint(0, 2, (n_samples, n_classes))
y_score = np.random.rand(n_samples, n_classes)

plt.figure(figsize=(12, 8))

# Colors for different classes
colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2, 
             label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for 16 Fault Classes', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", ncol=2, fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ ROC curves saved as: roc_curves.png")