import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, confusion_matrix
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

# -----------------------------
# ✅ 1. Check GPU Status
# -----------------------------
print("✅ Checking GPU status...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"✅ {len(gpus)} physical GPUs, {len(logical_gpus)} logical GPUs available")
    except RuntimeError as e:
        print("❌ GPU configuration error:", e)
else:
    print("❌ No GPU detected, using CPU (slower performance)")

# -----------------------------
# ✅ 2. Create Test Dataset
# -----------------------------
def test_image_generator(folder, target_size=(224, 224)):
    """
    Generator function to yield images and labels from folder.
    Assumes filenames are in format: label_filename.png
    """
    filenames = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        try:
            img = tf.keras.utils.load_img(img_path, target_size=target_size)
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            label = int(filename.split('_')[0])  # Extract label from filename
            yield img_array, label
        except Exception as e:
            print(f"⚠️ Skipping {filename}: {e}")
            continue


def create_test_dataset(folder, batch_size=8, img_size=(224, 224)):
    """
    Create a tf.data.Dataset from a folder of test images using a generator.
    Adds a time dimension for LSTM compatibility.
    """
    output_signature = (
        tf.TensorSpec(shape=(*img_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: test_image_generator(folder, img_size),
        output_signature=output_signature
    )
    dataset = dataset.batch(batch_size)
    # Add time dimension: (B, H, W, C) -> (B, 1, H, W, C)
    dataset = dataset.map(
        lambda x, y: (tf.expand_dims(x, axis=1), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# -----------------------------
# ✅ 3. Load Model & Data
# -----------------------------
# Load trained model
model_path = r'E:\software\pycharm\workspace\multi-view\other_model\CNN_LSTM_H1_model.h5'
model = load_model(model_path)
print(f"✅ Model loaded from: {model_path}")

# Test data folder
test_folder = r'D:\Dataset\H1_O3a_data\evaluation'
print(f"📊 Loading test data from: {test_folder}")

# Create test dataset
test_dataset = create_test_dataset(test_folder, batch_size=8, img_size=(224, 224))

# Extract true labels
print("📌 Extracting true labels...")
y_true = [label.numpy() for _, label in test_dataset.unbatch()]
y_true = np.array(y_true)

# Perform prediction (GPU acceleration is automatic if available)
print("📌 Starting prediction...")
y_pred_probs = model.predict(test_dataset)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# -----------------------------
# ✅ 4. Save Results & Evaluation
# -----------------------------
# Save predictions to CSV
csv_path = r'E:\software\pycharm\workspace\multi-view\classification_results.csv'
df_results = pd.DataFrame({
    'Image': [f"Image_{i+1}" for i in range(len(y_true))],
    'True_Label': y_true,
    'Predicted_Label': y_pred_classes
})
df_results.to_csv(csv_path, index=False)
print(f"✅ Classification results saved to: {csv_path}")

# Calculate and print evaluation metrics
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")

# Optional: Full classification report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Generate and save confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=sorted(np.unique(y_true)),
    yticklabels=sorted(np.unique(y_true))
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
conf_matrix_path = r'E:\software\pycharm\workspace\multi-view\confusion_matrix.png'
plt.savefig(conf_matrix_path)
plt.show()
print(f"✅ Confusion matrix saved to: {conf_matrix_path}")

print("🎉 Evaluation completed successfully!")