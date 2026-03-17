"""
visualize.py — Show all 15 corruptions on sample letters.
Run this to check that corruptions look right before building the dataset.
"""

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from corruptions import *


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.transpose(image, perm=[1, 0, 2])
    label = label - 1
    return image, label


# Grab one image per class
ds = tfds.load('emnist/letters', split='test', as_supervised=True)
samples = {}
for img, lbl in ds.map(preprocess):
    l = int(lbl.numpy())
    if l not in samples:
        samples[l] = img.numpy().squeeze()
    if len(samples) >= 26:
        break

# Pick 8 from whatever we found
available = sorted(samples.keys())
np.random.seed(42)
pick = sorted(np.random.choice(available, size=min(8, len(available)), replace=False))
letters = [chr(65 + i) for i in range(26)]

corruptions = [
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

fig, axes = plt.subplots(len(pick), len(corruptions), figsize=(20, 10))
for row, idx in enumerate(pick):
    for col, (name, fn) in enumerate(corruptions):
        img_255 = (samples[idx] * 255).astype(np.float32)
        corrupted = np.clip(fn(img_255.copy()), 0, 255) / 255.0
        axes[row, col].imshow(corrupted, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        if row == 0:
            axes[row, col].set_title(name.replace('_', '\n'), fontsize=6)
        if col == 0:
            axes[row, col].set_ylabel(letters[idx], fontsize=11, rotation=0, labelpad=12)

plt.suptitle('EMNIST-C Corruption Samples', fontsize=14)
plt.tight_layout()
plt.savefig('corruption_samples.png', dpi=200, bbox_inches='tight')
plt.show()
print('Saved corruption_samples.png')