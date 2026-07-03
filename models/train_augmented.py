"""
train_augmented.py — Train an augmented CNN on EMNIST Letters.

What this script does (simple explanation):
  - It trains the same CNN as the tuned baseline
  - But during training, it also shows the model corrupted/noisy letters
  - This makes the model more robust to corruptions
  - The final model is saved as 'augmented_cnn.keras'

How to run:
  python train_augmented.py

How long does it take?
  - About 20-40 minutes depending on your computer
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import json
import random

# ── Best settings found by Optuna (from tune_baseline.py) ───────────────────

FILTERS1      = 32
FILTERS2      = 128
DENSE_UNITS   = 128
DROPOUT       = 0.5
LEARNING_RATE = 0.002643803942855385
BATCH_SIZE    = 256
EPOCHS        = 20

# ── Data loading ─────────────────────────────────────────────────────────────

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.transpose(image, perm=[1, 0, 2])  # EMNIST is stored transposed
    label = label - 1                             # Labels 1-26 → 0-25
    return image, label


print("Loading data...")
train_ds_raw = tfds.load('emnist/letters', split='train', as_supervised=True)
test_ds_raw  = tfds.load('emnist/letters', split='test',  as_supervised=True)
print("Data loaded.\n")

# ── Augmentation functions ───────────────────────────────────────────────────
#
# These are applied randomly during training.
# Each image has a chance to be corrupted in one of these ways.
# This teaches the model to handle messy input.
#
# We use the same corruption families as corruptions.py:
#   - rotation
#   - blur
#   - noise
#   - brightness
#   - shift (translate)

def augment(image, label):
    """
    Randomly apply one or more augmentations to a single image.
    All operations work on float32 images in range [0, 1].
    """

    # 1. Random rotation (up to 20 degrees left or right)
    if tf.random.uniform(()) > 0.5:
        angle = tf.random.uniform((), -0.35, 0.35)  # radians ≈ 20 degrees
        image = tfa_rotate(image, angle)

    # 2. Random blur (Gaussian blur effect via depthwise conv)
    if tf.random.uniform(()) > 0.5:
        image = random_blur(image)

    # 3. Random noise (salt and pepper style)
    if tf.random.uniform(()) > 0.5:
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
        image = tf.clip_by_value(image + noise, 0.0, 1.0)

    # 4. Random brightness shift
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.clip_by_value(image, 0.0, 1.0)

    # 5. Random shift (translate) left/right/up/down
    if tf.random.uniform(()) > 0.5:
        image = random_shift(image)

    return image, label


def tfa_rotate(image, angle):
    """Rotate image by angle (radians) using affine transform."""
    # Build rotation matrix
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)
    # Center of 28x28 image is (14, 14)
    cx, cy = 14.0, 14.0
    # Translate → rotate → translate back
    transform = [
        cos_a, -sin_a, cx - cx * cos_a + cy * sin_a,
        sin_a,  cos_a, cy - cx * sin_a - cy * cos_a,
        0.0, 0.0
    ]
    transform = tf.reshape(tf.cast(transform, tf.float32), [1, 8])
    image_4d = tf.expand_dims(image, 0)  # [1, 28, 28, 1]
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image_4d,
        transforms=transform,
        output_shape=tf.constant([28, 28]),
        interpolation='BILINEAR',
        fill_mode='CONSTANT',
        fill_value=0.0,
    )
    return tf.squeeze(rotated, 0)


def random_blur(image):
    """Apply a simple blur by averaging with a shifted version."""
    # Simple box blur approximation using average pooling
    image_4d = tf.expand_dims(image, 0)  # [1, 28, 28, 1]
    blurred = tf.nn.avg_pool2d(image_4d, ksize=3, strides=1, padding='SAME')
    return tf.squeeze(blurred, 0)


def random_shift(image):
    """Shift image randomly by up to 3 pixels in any direction."""
    # Pad then crop to simulate shift
    pad = 3
    image_4d = tf.expand_dims(image, 0)
    padded = tf.pad(image_4d, [[0,0],[pad,pad],[pad,pad],[0,0]])
    dx = tf.random.uniform((), 0, 2*pad, dtype=tf.int32)
    dy = tf.random.uniform((), 0, 2*pad, dtype=tf.int32)
    cropped = padded[:, dy:dy+28, dx:dx+28, :]
    cropped = tf.ensure_shape(cropped, [1, 28, 28, 1])
    return tf.squeeze(cropped, 0)


# ── Build datasets ────────────────────────────────────────────────────────────
#
# Training data: preprocessed + augmented (corrupted versions mixed in)
# Test data:     preprocessed only (clean, no augmentation)

train_ds = (train_ds_raw
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .shuffle(10000)
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)  # ← augmentation here
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))

test_ds = (test_ds_raw
           .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
           .cache()
           .batch(BATCH_SIZE)
           .prefetch(tf.data.AUTOTUNE))

# ── Build model (same architecture as tuned baseline) ────────────────────────

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # First convolutional block
    layers.Conv2D(FILTERS1, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # Second convolutional block
    layers.Conv2D(FILTERS2, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    # Dense layer with dropout
    layers.Dense(DENSE_UNITS, activation='relu'),
    layers.Dropout(DROPOUT),

    # Output: 26 classes (A-Z)
    layers.Dense(26, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────

print("\nTraining augmented model...")
print("This will take about 20-40 minutes. Go grab a coffee! ☕\n")

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    verbose=1,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────

loss, acc = model.evaluate(test_ds, verbose=0)
print(f"\nAugmented model — Clean test accuracy: {acc * 100:.2f}%")

# Save the model
model.save('augmented_cnn.keras')
print("Saved to augmented_cnn.keras")

# Save results to JSON
results = {
    'clean_accuracy': float(acc),
    'architecture': {
        'filters1': FILTERS1,
        'filters2': FILTERS2,
        'dense_units': DENSE_UNITS,
        'dropout': DROPOUT,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
    },
    'augmentations_used': [
        'random_rotation (up to 20 degrees)',
        'random_blur (box blur)',
        'random_gaussian_noise (stddev=0.05)',
        'random_brightness (delta=0.2)',
        'random_shift (up to 3 pixels)',
    ]
}
with open('augmented_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved augmented_results.json")

# ── Plot training history ─────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'],     label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Augmented Model — Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history.history['loss'],     label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Augmented Model — Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig('augmented_training_history.png', dpi=150)
plt.show()
print("Saved augmented_training_history.png")

print("\n✅ All done!")
print("Now run evaluate.py to compare baseline vs augmented:")
print("  python evaluate.py --compare augmented_cnn.keras")
