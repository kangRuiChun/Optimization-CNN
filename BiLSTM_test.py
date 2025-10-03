import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Compatibility for older NumPy versions
if not hasattr(np, 'bool_'):
    np.bool_ = np.dtype('bool')
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict

# -------------------------------
# 0. GPU Configuration Check
# -------------------------------
print("🔍 Checking GPU status...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"✅ {len(gpus)} physical GPU(s), {len(logical_gpus)} logical GPU(s) available.")
    except RuntimeError as e:
        print("❌ GPU configuration error:", e)
else:
    print("⚠️ No GPU detected, inference will run on CPU (slower)")

# -------------------------------
# 1. Test Dataset Generator (Efficient Loading + GPU Inference)
# -------------------------------
def create_test_dataset(base_root, time_dirs, subset='evaluation', img_size=(224, 224), batch_size=8):
    """
    Creates a tf.data.Dataset for efficient testing (supports GPU batch processing)
    Returns:
        dataset: tf.data.Dataset object for batched inference
        true_labels: List of ground truth labels (for evaluation)
    """
    first_time_dir = os.path.join(base_root, time_dirs[0], subset)
    if not os.path.exists(first_time_dir):
        raise FileNotFoundError(f"Subset directory not found: {first_time_dir}")

    filenames = sorted([f for f in os.listdir(first_time_dir) if f.lower().endswith('.png')])
    file_list = []

    print(f"🔍 Scanning '{subset}' subset, found {len(filenames)} candidate files...")

    for fname in filenames:
        try:
            label = int(fname.split('_')[0])
            if label < 0 or label >= 22:  # Only keep labels in [0, 22)
                continue
            all_exist = True
            paths = []
            for t_dir in time_dirs:
                img_path = os.path.join(base_root, t_dir, subset, fname)
                if not os.path.exists(img_path):
                    all_exist = False
                    break
                paths.append(img_path)
            if all_exist:
                file_list.append((paths, label))
        except Exception as e:
            print(f"[Skipped] Error parsing {fname}: {e}")
            continue

    print(f"✅ Found {len(file_list)} valid samples")

    def generator():
        for paths, label in file_list:
            image_sequence = []
            try:
                for img_path in paths:
                    img = load_img(img_path, target_size=img_size)
                    img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
                    image_sequence.append(img_array)
                yield np.array(image_sequence), label
            except Exception as e:
                print(f"❌ Error loading image sequence {paths}: {e}")
                # Return placeholder to avoid pipeline interruption
                dummy_seq = np.zeros((len(time_dirs), *img_size, 3), dtype=np.float32)
                yield dummy_seq, 0

    # Build dataset using tf.data for performance
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(len(time_dirs), *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    # Batch and prefetch for improved GPU utilization
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, [label for _, label in file_list]  # Return dataset and true labels


# -------------------------------
# 2. Main Testing Program
# -------------------------------
if __name__ == "__main__":
    # --- Configuration ---
    BASE_ROOT = r'D:\Dataset\H1_O3a_data\LSTM\LSTM_after'  # Ensure path consistency
    MODEL_PATH = r'E:\software\pycharm\workspace\multi-view\other_model\best_cnn_bilstm_model.h5'
    TIME_DIRS = ['0.5s', '1s', '2s', '4s']
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 8  # Must match training configuration

    # --- Load Trained Model ---
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print(f"🔄 Loading trained model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # --- Build Test Dataset ---
    print("🚀 Building test dataset...")
    test_dataset, true_labels = create_test_dataset(
        base_root=BASE_ROOT,
        time_dirs=TIME_DIRS,
        subset='evaluation',
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    if len(true_labels) == 0:
        raise ValueError("❌ Test set is empty. Please check data paths or filename formatting!")

    print(f"✅ Test dataset built successfully with {len(true_labels)} samples")

    # --- GPU-Accelerated Prediction ---
    print("🔮 Performing batch predictions using GPU...")
    predictions = model.predict(test_dataset, verbose=1)  # Automatically uses GPU if available
    predicted_classes = np.argmax(predictions, axis=1)

    # --- Compute Evaluation Metrics ---
    acc = accuracy_score(true_labels, predicted_classes)
    print(f"\n📊 Test Accuracy: {acc:.4f}")

    print("\n📋 Classification Report (Precision, Recall, F1-score):")
    report = classification_report(true_labels, predicted_classes, digits=4)
    print(report)

    # --- Confusion Matrix ---
    cm = confusion_matrix(true_labels, predicted_classes)
    num_classes = cm.shape[0]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=range(num_classes),
                yticklabels=range(num_classes))
    plt.title(f'Confusion Matrix\nAccuracy: {acc:.4f}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

    # Save classification report to file
    save_dir = "../other_model"
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)

    print(f"✅ Confusion matrix saved: confusion_matrix.png")
    print(f"✅ Classification report saved: {report_path}")
    print("🎉 Testing completed! GPU-accelerated inference executed successfully.")