import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import pandas as pd

# Fix NumPy deprecation issues
if not hasattr(np, 'bool_'):
    np.bool_ = np.dtype('bool')
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict


def load_images_from_folder(folder, target_size=(224, 224)):
    """
    Load images from a folder and generate labels from filenames.
    Assumes filename format: label_filename.png
    """
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize to [0,1]
            images.append(img_array)
            label = int(filename.split('_')[0])  # Extract label from filename
            labels.append(label)
    return np.array(images), np.array(labels)


# Data path
test_folder = r'D:\Dataset\H1_O3a_data\evaluation'  # Update to your test data folder

# Load the pre-trained model
model = load_model(r'E:\software\pycharm\workspace\multi-view\package\Early_Fusion_model.h5')
print("✅ Model loaded successfully.")

# Load test data
print("📌 Loading test images...")
test_images, test_labels = load_images_from_folder(test_folder)
print(f"✅ Loaded {len(test_images)} test images.")

# Make predictions
print("📌 Performing predictions...")
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate evaluation metrics
accuracy = accuracy_score(test_labels, y_pred_classes)
precision = precision_score(test_labels, y_pred_classes, average='weighted')
f1 = f1_score(test_labels, y_pred_classes, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Plot and save confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=np.unique(test_labels),
    yticklabels=np.unique(test_labels)
)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
conf_matrix_path = r'E:\software\pycharm\workspace\multi-view\confusion-L1_matrix.png'
plt.savefig(conf_matrix_path)
plt.show()
print(f"✅ Confusion matrix saved to: {conf_matrix_path}")

# Classification report
class_report = classification_report(test_labels, y_pred_classes)
print("Classification Report:")
print(class_report)

# Compute AUC for multi-class using one-hot encoding
label_binarizer = LabelBinarizer()
y_true_one_hot = label_binarizer.fit_transform(test_labels)
y_pred_one_hot = y_pred  # Model outputs are already probabilities

num_classes = y_true_one_hot.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute ROC curve and AUC for each class
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_one_hot[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for all classes
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# Add diagonal line
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

# Formatting
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("✅ ROC curve displayed.")

# Optional: Micro-average ROC and AUC
fpr_micro, tpr_micro, _ = roc_curve(y_true_one_hot.ravel(), y_pred_one_hot.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

plt.figure(figsize=(10, 8))
plt.plot(fpr_micro, tpr_micro, label=f'Micro-average ROC (AUC = {roc_auc_micro:.2f})',
         color='deeppink', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Micro-average ROC Curve')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)
plt.show()

print(f"✅ Micro-average AUC: {roc_auc_micro:.4f}")
print("🎉 Evaluation completed successfully!")