import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt

# -------------------------------
# 1. GPU Configuration and Mixed Precision
# -------------------------------
print("🔍 Checking GPU status...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"✅ {len(gpus)} physical GPUs, {len(logical_gpus)} logical GPUs available.")
    except RuntimeError as e:
        print("❌ GPU configuration error:", e)
else:
    print("❌ No GPU detected, using CPU (training will be extremely slow)")
    exit()

# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print(f"✅ Mixed precision enabled: {tf.keras.mixed_precision.global_policy()}")


def build_file_list(folder):
    """Scan folder and return a list of (filename, label) tuples"""
    if not os.path.exists(folder):
        raise FileNotFoundError(f"❌ Folder does not exist: {folder}")

    file_list = []
    for fname in os.listdir(folder):
        if fname.lower().endswith('.png'):
            try:
                label = int(fname.split('_')[0])
                if 0 <= label < 22:
                    file_list.append((fname, label))
            except:
                continue
    if len(file_list) == 0:
        raise ValueError(f"❌ No valid samples in {folder}, please check naming format (label_filename.png)")
    return file_list


# -------------------------------
# 3. Create Efficient tf.data Dataset
# -------------------------------
def create_dataset_from_paths(folder, file_list, img_size=(224, 224), batch_size=4, shuffle=False, repeat=True):
    def parse_image(filename, label):
        filepath = folder + '/' + filename
        img = tf.io.read_file(filepath)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    # Create dataset from filenames and labels
    filenames = [f[0] for f in file_list]
    labels = [f[1] for f in file_list]

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filenames), seed=42)

    # Decode images (multi-threaded)
    dataset = dataset.map(
        parse_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Add time_steps dimension: (B, H, W, C) -> (B, 1, H, W, C)
    dataset = dataset.map(
        lambda x, y: (tf.expand_dims(x, axis=1), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Repeat for multiple epochs
    if repeat:
        dataset = dataset.repeat()

    # Prefetch to overlap data loading and training
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_cnn_mp_lstm_model(input_shape, num_classes, time_steps=1):
    input_layer = layers.Input(shape=(time_steps,) + input_shape)

    x = input_layer
    filters = 32
    for _ in range(4):
        x = layers.TimeDistributed(
            layers.Conv2D(filters, (3, 3), activation='relu', padding='same')
        )(x)
        x = layers.TimeDistributed(
            layers.MaxPooling2D((2, 2))
        )(x)
        filters *= 2

    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.LSTM(512, return_sequences=False)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model


if __name__ == "__main__":
    TRAIN_FOLDER = r'D:\Dataset\H1_O3a_data\training'
    VAL_FOLDER = r'D:\Dataset\H1_O3a_data\validation'
    MODEL_SAVE_PATH = r'E:\software\pycharm\workspace\multi-view\other_model\CNN_LSTM_H1_model.h5'
    LOG_DIR = r'E:\software\pycharm\workspace\multi-view\logs'

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 4
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 22

    # --- 1. Build file-path-label list ---
    print("📊 Scanning training set...")
    train_list = build_file_list(TRAIN_FOLDER)
    print("📊 Scanning validation set...")
    val_list = build_file_list(VAL_FOLDER)

    num_train = len(train_list)
    num_val = len(val_list)

    steps_per_epoch = num_train // BATCH_SIZE
    validation_steps = num_val // BATCH_SIZE

    print(f"✅ Training samples: {num_train} → steps_per_epoch: {steps_per_epoch}")
    print(f"✅ Validation samples: {num_val} → validation_steps: {validation_steps}")

    if steps_per_epoch == 0:
        raise ValueError("⚠️ steps_per_epoch is 0, please reduce BATCH_SIZE")
    if validation_steps == 0:
        print("⚠️ validation_steps is 0, consider increasing validation set or reducing BATCH_SIZE")

    # --- 2. Create datasets ---
    train_dataset = create_dataset_from_paths(
        TRAIN_FOLDER, train_list,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        repeat=True  # 🔁 Must repeat for training
    )

    val_dataset = create_dataset_from_paths(
        VAL_FOLDER, val_list,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        repeat=False  # ❌ Do not repeat validation set
    )

    # --- 3. Test data pipeline ---
    try:
        sample_batch = next(iter(train_dataset))
        print(f"✅ Data pipeline test passed, input shape: {sample_batch[0].shape}")
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        exit()

    # --- 4. Build model ---
    model = create_cnn_mp_lstm_model(
        input_shape=IMG_SIZE + (3,),
        num_classes=NUM_CLASSES,
        time_steps=1
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # --- 5. Callbacks ---
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_graph=True, write_images=True)
    ]

    # --- 6. Start training ---
    print("🚀 Starting training (stable mode, GPU + mixed precision)...")

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # --- 7. Plot training curves ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"🎉 Training completed! Model saved to: {MODEL_SAVE_PATH}")