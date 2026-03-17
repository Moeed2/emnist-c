"""
build_emnist_c.py — Apply all 15 corruptions to the EMNIST Letters test set.
Saves each as a .npy file in emnist_c/.
"""

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import time
from corruptions import (
    identity, shot_noise, impulse_noise, gaussian_blur, motion_blur,
    shear, scale, rotate, brightness, translate,
    stripe, fog, spatter, dotted_line, zigzag, canny_edges
)

CORRUPTIONS = [
    ('identity', identity),
    ('shot_noise', shot_noise),
    ('impulse_noise', impulse_noise),
    ('gaussian_blur', gaussian_blur),
    ('motion_blur', motion_blur),
    ('shear', shear),
    ('scale', scale),
    ('rotate', rotate),
    ('brightness', brightness),
    ('translate', translate),
    ('stripe', stripe),
    ('fog', fog),
    ('spatter', spatter),
    ('dotted_line', dotted_line),
    ('zigzag', zigzag),
    ('canny_edges', canny_edges),
]


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.transpose(image, perm=[1, 0, 2])
    label = label - 1
    return image, label


# Load test set
print("Loading test set...")
ds = tfds.load('emnist/letters', split='test', as_supervised=True)
images, labels = [], []
for img, lbl in ds.map(preprocess):
    images.append(img.numpy())
    labels.append(lbl.numpy())
x_clean = np.array(images)
y_test = np.array(labels)
print(f"Test set: {len(x_clean)} images")

# Save
os.makedirs('emnist_c', exist_ok=True)
np.save('emnist_c/labels.npy', y_test)

for name, fn in CORRUPTIONS:
    t0 = time.time()
    print(f"  {name}...", end=' ', flush=True)
    corrupted = []
    for img in x_clean:
        img_255 = (img.squeeze() * 255.0).astype(np.float32)
        c = fn(img_255)
        corrupted.append(np.clip(c, 0, 255).astype(np.float32)[..., np.newaxis] / 255.0)
    np.save(f'emnist_c/{name}.npy', np.array(corrupted, dtype=np.float32))
    print(f"done ({time.time() - t0:.1f}s)")

print(f"\nSaved {len(CORRUPTIONS)} files + labels to emnist_c/")
