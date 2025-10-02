import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import pandas as pd

# Address NumPy deprecation issues
if not hasattr(np, 'bool_'):
    np.bool_ = np.dtype('bool')

if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict


# Load images from folder and generate labels
def load_images_from_folder(folder, target_size=(224, 224)):
    """
    Load images from a directory, resize to target size, normalize pixel values,
    and extract labels from filenames.
    """
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize to [0,1]
            images.append(img_array)
            label = int(filename.split('_')[0])  # Extract class label from filename
            labels.append(label)
    return np.array(images), np.array(labels)


# Data path
test_folder = r'D:\Dataset\L1_O3b_data\evaluation'  # Path to the test dataset

# Load the pre-trained model
model = load_model(r'E:\software\pycharm\workspace\multi-view\package\GravitySpy_H1_model2.h5')

# Load test data
test_images, test_labels = load_images_from_folder(test_folder)

# Make predictions
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Get list of filenames
filenames = [filename for filename in os.listdir(test_folder) if filename.lower().endswith('.png')]

# Calculate accuracy, precision, and F1 score
accuracy = accuracy_score(test_labels, y_pred_classes)
precision = precision_score(test_labels, y_pred_classes, average='weighted')
f1 = f1_score(test_labels, y_pred_classes, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification report
class_report = classification_report(test_labels, y_pred_classes)
print("Classification Report:")
print(class_report)

# Compute AUC for multi-class classification
label_binarizer = LabelBinarizer()
y_true_one_hot = label_binarizer.fit_transform(test_labels)
y_pred_one_hot = y_pred

num_classes = y_true_one_hot.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_one_hot[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()