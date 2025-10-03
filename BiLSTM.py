import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Compatibility for older NumPy versions
if not hasattr(np, 'bool_'):
    np.bool_ = np.dtype('bool')
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict

# -------------------------------
# 1. Spatiotemporal Data Generator
# -------------------------------
def spatiotemporal_generator(base_root, time_dirs, subset, img_size=(224, 224), batch_size=8):
    """
    Generator: Dynamically loads spatiotemporal sequence data in batches
    Yields: (batch_data, batch_labels)
    """
    first_time_dir = os.path.join(base_root, time_dirs[0], subset)
    if not os.path.exists(first_time_dir):
        raise FileNotFoundError(f"Subset directory not found: {first_time_dir}")

    filenames = [f for f in os.listdir(first_time_dir) if f.lower().endswith('.png')]
    if len(filenames) == 0:
        raise ValueError(f"No PNG files found in {first_time_dir}")

    print(f"🔍 Found {len(filenames)} samples in {subset} set, loading with generator...")

    # Shuffle filenames (recommended only for training)
    if subset == 'training':
        np.random.shuffle(filenames)

    # Cache full paths and labels to avoid repeated checks
    file_list = []
    for fname in filenames:
        try:
            label = int(fname.split('_')[0])  # Extract label: '09_xxx.png' -> 9
        except Exception as e:
            print(f"[Skipped] Failed to parse label {fname}: {e}")
            continue

        # Validate label range [0, 22)
        if label < 0 or label >= 22:
            print(f"[Skipped] Label out of range [0,22): {fname} -> label={label}")
            continue

        all_exist = True
        paths = []
        for t_dir in time_dirs:
            img_path = os.path.join(base_root, t_dir, subset, fname)
            if not os.path.exists(img_path):
                print(f"[Skipped] Missing image: {img_path}")
                all_exist = False
                break
            paths.append(img_path)
        if all_exist:
            file_list.append((paths, label))

    print(f"✅ Valid samples in {subset} set: {len(file_list)}")
    if len(file_list) == 0:
        raise ValueError(f"❌ No valid samples in {subset} set. Please check file paths and integrity!")

    # Start generating batches
    idx = 0
    while True:
        batch_data = []
        batch_labels = []

        # Build one batch
        for _ in range(batch_size):
            if idx >= len(file_list):
                idx = 0
                if subset != 'training':  # Non-training sets iterate once
                    break

            paths, label = file_list[idx]
            image_sequence = []

            try:
                for img_path in paths:
                    img = load_img(img_path, target_size=img_size)
                    img_array = img_to_array(img) / 255.0  # Normalize
                    image_sequence.append(img_array)
                batch_data.append(image_sequence)
                batch_labels.append(label)
            except Exception as e:
                print(f"❌ Error loading image sequence {paths}: {e}")

            idx += 1

        if len(batch_data) == 0:
            continue  # Restart

        yield np.array(batch_data), np.array(batch_labels)


# -------------------------------
# 2. Build Model: 4-layer CNN + BiLSTM
# -------------------------------
def create_spatiotemporal_model(input_shape, num_classes, time_steps=4):
    input_layer = layers.Input(shape=(time_steps,) + input_shape)
    x = input_layer

    # 4 layers of CNN + MaxPooling (TimeDistributed)
    filters = 32
    for _ in range(4):
        x = layers.TimeDistributed(
            layers.Conv2D(filters, (3, 3), activation='relu', padding='same')
        )(x)
        x = layers.TimeDistributed(
            layers.MaxPooling2D((2, 2))
        )(x)
        filters *= 2  # 32 → 64 → 128 → 256

    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=False))(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model


# -------------------------------
# 3. Main Program
# -------------------------------
if __name__ == "__main__":
    # --- Configuration ---
    BASE_ROOT = r'D:\Dataset\H1_O3a_data\LSTM\LSTM_after'
    TIME_DIRS = ['0.5s', '1s', '2s', '4s']
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 22  # ✅ Set to 22 classes

    # --- Count samples for steps_per_epoch ---
    def count_samples(base_root, time_dirs, subset):
        first_dir = os.path.join(base_root, time_dirs[0], subset)
        if not os.path.exists(first_dir):
            return 0
        filenames = [f for f in os.listdir(first_dir) if f.lower().endswith('.png')]
        valid_count = 0
        for fname in filenames:
            try:
                label = int(fname.split('_')[0])
                if label < 0 or label >= 22:
                    continue  # Skip invalid labels
            except:
                continue

            all_exist = True
            for t_dir in time_dirs:
                if not os.path.exists(os.path.join(base_root, t_dir, subset, fname)):
                    all_exist = False
                    break
            if all_exist:
                valid_count += 1
        return valid_count

    train_steps = count_samples(BASE_ROOT, TIME_DIRS, 'training') // BATCH_SIZE
    val_steps = max(1, count_samples(BASE_ROOT, TIME_DIRS, 'validation') // BATCH_SIZE)
    test_steps = max(1, count_samples(BASE_ROOT, TIME_DIRS, 'evaluation') // BATCH_SIZE)

    if train_steps == 0:
        raise ValueError("❌ No training samples found. Please check data paths and file integrity!")

    print(f"📊 Training steps: {train_steps}, Validation steps: {val_steps}, Test steps: {test_steps}")

    # --- Create tf.data.Dataset ---
    train_gen = spatiotemporal_generator(BASE_ROOT, TIME_DIRS, 'training', IMG_SIZE, BATCH_SIZE)
    val_gen = spatiotemporal_generator(BASE_ROOT, TIME_DIRS, 'validation', IMG_SIZE, BATCH_SIZE)
    test_gen = spatiotemporal_generator(BASE_ROOT, TIME_DIRS, 'evaluation', IMG_SIZE, BATCH_SIZE)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, len(TIME_DIRS), *IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).repeat().prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: val_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, len(TIME_DIRS), *IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).repeat().prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, len(TIME_DIRS), *IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).repeat(1).prefetch(tf.data.AUTOTUNE)  # Evaluation runs once

    # --- Build Model ---
    model = create_spatiotemporal_model(
        input_shape=IMG_SIZE + (3,),
        num_classes=NUM_CLASSES,
        time_steps=len(TIME_DIRS)
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # --- Callbacks ---
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=r'E:\software\pycharm\workspace\multi-view\other_model\best_cnn_bilstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # --- Training ---
    print("🚀 Starting training...")
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )

    # --- Save Model ---
    model.save('final_cnn_bilstm_model.h5')
    print("✅ Model saved: final_cnn_bilstm_model.h5")

    # --- Evaluate on Test Set ---
    print("📊 Evaluating on test set...")
    test_results = model.evaluate(test_dataset, steps=test_steps, verbose=1)
    print(f"✅ Test Accuracy: {test_results[1]:.4f}, Loss: {test_results[0]:.4f}")

    # --- Plot Training Curves ---
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    print("🎉 Training and evaluation completed!")