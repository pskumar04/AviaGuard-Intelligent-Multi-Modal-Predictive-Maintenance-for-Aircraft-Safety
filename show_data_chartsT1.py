# show_data_chartsT1.py - COMPLETE WORKING VERSION
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns

# Skip ADASYN if not available (use simple random oversampling instead)
try:
    from imblearn.over_sampling import ADASYN
    adasyn_available = True
except ImportError:
    adasyn_available = False
    print("⚠️ ADASYN not installed, using random oversampling for demo")

# Generate sample data
np.random.seed(42)
n_samples = 1000
n_features = 50

# Create imbalanced dataset (80% normal, 20% faults)
X = np.random.randn(n_samples, n_features)
y = np.array([0]*800 + [1]*100 + [2]*60 + [3]*40)  # 4 classes

# 1. BEFORE - Class Distribution Chart
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
before_counts = Counter(y)
classes = list(before_counts.keys())
counts = list(before_counts.values())
colors = ['blue', 'orange', 'green', 'red']
bars = plt.bar(classes, counts, color=colors[:len(classes)], alpha=0.7)
plt.title('📊 BEFORE: Imbalanced Classes', fontsize=14, fontweight='bold')
plt.xlabel('Fault Class', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(classes, ['Normal', 'Fault Type 1', 'Fault Type 2', 'Fault Type 3'])
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             str(count), ha='center', fontsize=11)

# 2. AFTER - Balanced Classes (using simple oversampling for demo)
if adasyn_available:
    # Use real ADASYN if available
    adasyn = ADASYN(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
else:
    # Simple random oversampling for demo
    from sklearn.utils import resample
    X_resampled = []
    y_resampled = []
    max_count = max(before_counts.values())
    for class_label in classes:
        class_indices = np.where(y == class_label)[0]
        X_class = X[class_indices]
        if len(class_indices) < max_count:
            # Oversample minority classes
            n_missing = max_count - len(class_indices)
            indices = np.random.choice(len(class_indices), n_missing, replace=True)
            X_oversampled = X_class[indices]
            X_resampled.extend(X_class)
            X_resampled.extend(X_oversampled)
            y_resampled.extend([class_label] * len(class_indices))
            y_resampled.extend([class_label] * n_missing)
        else:
            X_resampled.extend(X_class)
            y_resampled.extend([class_label] * len(class_indices))
    
    X_resampled = np.array(X_resampled)
    y_resampled = np.array(y_resampled)

plt.subplot(1, 3, 2)
after_counts = Counter(y_resampled)
classes_after = list(after_counts.keys())
counts_after = list(after_counts.values())
bars = plt.bar(classes_after, counts_after, color=colors[:len(classes_after)], alpha=0.7)
plt.title('✅ AFTER: Balanced Classes', fontsize=14, fontweight='bold')
plt.xlabel('Fault Class', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(classes_after, ['Normal', 'Fault Type 1', 'Fault Type 2', 'Fault Type 3'])
for bar, count in zip(bars, counts_after):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             str(count), ha='center', fontsize=11)

# 3. PCA Variance Explained
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_resampled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.subplot(1, 3, 3)
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'b-o', linewidth=2, markersize=4)
plt.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95% Variance Threshold')
plt.title('📈 PCA: Cumulative Variance Explained', fontsize=14, fontweight='bold')
plt.xlabel('Number of Principal Components', fontsize=12)
plt.ylabel('Cumulative Variance Ratio', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Find number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
plt.text(n_components_95, 0.5, f' {n_components_95} components\n for 95% variance', 
         fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()
plt.savefig('jagadeesh_data_preprocessing.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("✅ DATA PREPROCESSING RESULTS")
print("="*50)
print(f"Original samples: {n_samples}")
print(f"Original features: {n_features}")
print(f"Original class distribution: {dict(before_counts)}")
print(f"Balanced class distribution: {dict(after_counts)}")
print(f"Components needed for 95% variance: {n_components_95}")
print(f"Feature reduction: {n_features} → {n_components_95} ({(1 - n_components_95/n_features)*100:.1f}% reduction)")
print("="*50)
print("✅ Chart saved as: jagadeesh_data_preprocessing.png")