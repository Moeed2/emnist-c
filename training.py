import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


(ds_train, ds_test), ds_info = tfds.load(
    "emnist/byclass",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

# Keep uppercase A–Z (10-35)
def filter_uppercase(image, label):
    return tf.logical_and(label >= 10, label <= 35)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)
    label = label - 10
    return image, label

BATCH_SIZE = 64

ds_train = (
    ds_train
    .filter(filter_uppercase)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

ds_test = (
    ds_test
    .filter(filter_uppercase)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# code for previewing images 

# sample_ds = ds_train.unbatch().take(25)
#
# def label_to_char(label):
#     return chr(label + ord('A'))
#
# plt.figure(figsize=(6, 6))
# for i, (image, label) in enumerate(sample_ds):
#     plt.subplot(5, 5, i + 1)
#     plt.imshow(tf.squeeze(image), cmap="gray")
#     plt.title(label_to_char(label.numpy()))
#     plt.axis("off")
#
# plt.tight_layout()
# plt.savefig("preview.png", dpi=200)

# Baseline CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test
)

test_loss, test_acc = model.evaluate(ds_test)
print("Test accuracy:", test_acc)

model.save("emnist_uppercase_baseline.keras")
print("Model saved succesfully.")