import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import time

start = time.time()

# Custom Data Generator Class
class ImageDataGeneratorFromFolder(Sequence):
    def __init__(self, folder_path, batch_size, target_size=(224, 224), shuffle=True):
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.filenames = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
        self.labels = [int(f.split('_')[0]) for f in self.filenames]
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        for filename in batch_x:
            img = load_img(os.path.join(self.folder_path, filename), target_size=self.target_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)

        return np.array(images), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(self.filenames)
            np.random.seed(42)
            np.random.shuffle(self.labels)


# Modified CNN model structure
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # First convolution and pooling layer
    model.add(layers.Conv2D(16, (5, 5), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    # Second convolution and pooling layer
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    # Third convolution and pooling layer
    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    # Fourth convolution and pooling layer
    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    # Flatten layer
    model.add(layers.Flatten())

    # Fully connected layer
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


# Custom callback to save loss per epoch
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


# Paths to data folders
train_folder = r'D:\Dataset\H1_O3b_data\training'
val_folder = r'D:\Dataset\H1_O3b_data\validation'

batch_size = 4

# Create generator instances
train_generator = ImageDataGeneratorFromFolder(train_folder, batch_size=batch_size)
validation_generator = ImageDataGeneratorFromFolder(val_folder, batch_size=batch_size)

# Get number of classes
num_classes = len(set(train_generator.labels))

# Build the model
model = create_cnn_model((224, 224, 3), num_classes)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Training parameters
epochs = 20

print("Starting training")

history_callback = LossHistory()
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[history_callback],
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator)
)

# Save the model
model.save(r'E:\software\pycharm\workspace\multi-view\package\GravitySpy_H1_model2.h5')

# Plot loss curves
plt.figure(figsize=(10, 8))
plt.plot(history_callback.train_losses, label='Training Loss')
plt.plot(history_callback.val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

end = time.time()
spend_time = end - start
print("Time spent:", spend_time)