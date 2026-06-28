# show_training_curves.py
import matplotlib.pyplot as plt
import numpy as np
import json

# Simulate training history (replace with your actual training history)
# During actual training, Keras returns history object
epochs = 50
history = {
    'accuracy': 0.75 + 0.004 * np.arange(50) + 0.05 * np.random.randn(50),
    'val_accuracy': 0.73 + 0.004 * np.arange(50) + 0.05 * np.random.randn(50),
    'loss': 0.8 - 0.01 * np.arange(50) + 0.02 * np.random.randn(50),
    'val_loss': 0.82 - 0.01 * np.arange(50) + 0.02 * np.random.randn(50)
}

# Smooth the curves
for key in history:
    history[key] = np.convolve(history[key], np.ones(5)/5, mode='valid')

plt.figure(figsize=(15, 5))

# Accuracy curves
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
plt.plot(history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
plt.title('📈 Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0.7, 1.0])

# Add final accuracy text
final_acc = history['val_accuracy'][-1]
plt.text(45, final_acc-0.02, f'Final: {final_acc:.3f}', fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# Loss curves
plt.subplot(1, 2, 2)
plt.plot(history['loss'], 'b-', linewidth=2, label='Training Loss')
plt.plot(history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
plt.title('📉 Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Training curves saved as 'training_curves.png'")
print(f"Best Validation Accuracy: {max(history['val_accuracy']):.3f}")
print(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.3f}")