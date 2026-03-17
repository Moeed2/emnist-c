"""
baseline.py — Train a simple CNN on clean EMNIST Letters.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.transpose(image, perm=[1, 0, 2])  # EMNIST is stored transposed
    label = label - 1  # 1-26 → 0-25
    return image, label


# Load data
train_ds = tfds.load('emnist/letters', split='train', as_supervised=True)
test_ds = tfds.load('emnist/letters', split='test', as_supervised=True)

train_ds = train_ds.map(preprocess).cache().shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).cache().batch(128).prefetch(tf.data.AUTOTUNE)

# Model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=20, validation_data=test_ds)

loss, acc = model.evaluate(test_ds, verbose=0)
print(f"\nClean test accuracy: {acc * 100:.2f}%")

model.save('baseline_cnn.keras')
print("Saved to baseline_cnn.keras")

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()
print("Saved training_history.png")