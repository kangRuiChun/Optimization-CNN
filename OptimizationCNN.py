import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time

start = time.time()

# Fix NumPy deprecation issues
if not hasattr(np, 'bool_'):
    np.bool_ = np.dtype('bool')
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict


# Define CNN model
def create_cnn_model(input_shape, num_classes):
    """
    Creates a CNN model for image classification.
    """
    model = models.Sequential()
    # First convolutional and pooling layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    # Second convolutional and pooling layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    # Third convolutional and pooling layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    # Fourth convolutional and pooling layer
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    # Flatten layer
    model.add(layers.Flatten())
    # Fully connected layer
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


# Data paths
train_folder = r'D:\Dataset\H1_O3a_data\training'  # Update to your training data folder
val_folder = r'D:\Dataset\H1_O3a_data\validation'  # Update to your validation data folder


# Create dataset using tf.data
def create_dataset(image_paths, labels, batch_size=16, shuffle=True):
    """
    Creates a tf.data.Dataset from image paths and labels.
    Images are decoded, resized to 224x224, and normalized.
    """
    def _load_and_preprocess_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image /= 255.0  # Normalize to [0,1]
        return image, label

    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths))

    ds = ds.map(_load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


# Get image paths and labels from folder
def get_image_paths_and_labels(folder):
    """
    Scans folder for PNG images and extracts labels from filenames.
    Assumes filename format: label_filename.png
    """
    image_paths = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(folder, filename)
            image_paths.append(img_path)
            label = int(filename.split('_')[0])  # Extract label from filename
            labels.append(label)
    return image_paths, labels


# Load training and validation image paths and labels
train_image_paths, train_labels = get_image_paths_and_labels(train_folder)
val_image_paths, val_labels = get_image_paths_and_labels(val_folder)

# Check unique labels
print("Unique labels in training set:", np.unique(train_labels))
print("Unique labels in validation set:", np.unique(val_labels))

# Ensure labels are non-negative and start from 0
if np.min(train_labels) < 0:
    print("Warning: Negative labels found in the training set. Adjusting labels.")
    train_labels = [0 if label < 0 else label for label in train_labels]
if np.min(val_labels) < 0:
    print("Warning: Negative labels found in the validation set. Adjusting labels.")
    val_labels = [0 if label < 0 else label for label in val_labels]

# Convert to numpy arrays
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

# Determine number of classes
num_classes = len(np.unique(train_labels))

# Build datasets
train_ds = create_dataset(train_image_paths, train_labels, batch_size=16, shuffle=True)
val_ds = create_dataset(val_image_paths, val_labels, batch_size=16, shuffle=False)

# Build model
model = create_cnn_model((224, 224, 3), num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training parameters
epochs = 20

print("Starting training...")

# Train the model
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    verbose=1
)

# Save the trained model
model.save(r'E:\software\pycharm\workspace\multi-view\package\Early_Fusion_model.h5')
print("Model saved to: E:\\software\\pycharm\\workspace\\multi-view\\package\\Early_Fusion_model.h5")

# Plot training and validation loss
plt.figure(figsize=(10, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Calculate and print total training time
end = time.time()
spend_time = end - start
print(f"Total training time: {spend_time:.2f} seconds")