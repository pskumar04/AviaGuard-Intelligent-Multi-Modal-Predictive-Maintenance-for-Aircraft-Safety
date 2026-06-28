import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Sample data (replace with your actual predictions if available)
np.random.seed(42)
n_samples = 500
n_classes = 16

# Generate sample predictions (replace this with your actual model predictions)
y_true = np.random.randint(0, n_classes, n_samples)
y_pred = np.random.randint(0, n_classes, n_samples)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'F{i}' for i in range(n_classes)],
            yticklabels=[f'F{i}' for i in range(n_classes)],
            annot_kws={'size': 8})
plt.title('Multi-Class Confusion Matrix (16 Fault Types)', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Confusion matrix saved as: confusion_matrix.png")