"""
corruptions.py — 15 corruptions for EMNIST-C.

Inspired by MNIST-C (Mu & Gilmer, 2019). Our own implementation using
numpy and OpenCV. glass_blur replaced with gaussian_blur per CIFAR-10-C
(Hendrycks & Dietterich, 2018).

All functions: 28x28 float32 in [0,255] → 28x28 float32 in [0,255].
"""

import numpy as np
import cv2


def identity(x):
    return np.array(x, dtype=np.float32)


# ── Noise ─────────────────────────────────────────────────────────

def shot_noise(x, severity=3):
    c = [60, 25, 12, 5, 3][severity - 1]
    img = np.array(x, dtype=np.float64) / 255.0
    return (np.clip(np.random.poisson(img * c) / c, 0, 1) * 255).astype(np.float32)


def impulse_noise(x, severity=2):
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    img = np.array(x, dtype=np.float64) / 255.0
    mask = np.random.random(img.shape)
    img[mask < c / 2] = 0.0
    img[mask > 1 - c / 2] = 1.0
    return (img * 255).astype(np.float32)


# ── Blur ──────────────────────────────────────────────────────────

def gaussian_blur(x, severity=2):
    ksize = [3, 5, 7, 9, 11][severity - 1]
    return cv2.GaussianBlur(np.array(x, dtype=np.float32), (ksize, ksize), 0)


def motion_blur(x, severity=2):
    size = [3, 5, 7, 9, 11][severity - 1]
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0 / size
    angle = np.random.uniform(-45, 45)
    M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return np.clip(cv2.filter2D(np.array(x, dtype=np.float32), -1, kernel), 0, 255)


# ── Geometric ─────────────────────────────────────────────────────

def _warp(x, M):
    """Apply 2x3 affine matrix, keep image centered."""
    return cv2.warpAffine(np.array(x, dtype=np.float32), M, (28, 28),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def shear(x, severity=2):
    c = [0.2, 0.4, 0.6, 0.8, 1.0][severity - 1] * np.random.choice([-1, 1])
    M = np.float32([[1, c, -c * 14], [0, 1, 0]])
    return _warp(x, M)


def scale(x, severity=3):
    s = [0.9, 0.8, 0.7, 0.6, 0.5][severity - 1]
    M = cv2.getRotationMatrix2D((14, 14), 0, 1.0 / s)
    return _warp(x, M)


def rotate(x, severity=2):
    degrees = [11, 23, 34, 46, 57][severity - 1] * np.random.choice([-1, 1])
    M = cv2.getRotationMatrix2D((14, 14), degrees, 1.0)
    return _warp(x, M)


def translate(x, severity=3):
    c = [1, 2, 3, 4, 5][severity - 1]
    dx = c * np.random.choice([-1, 1])
    dy = c * np.random.choice([-1, 1])
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return _warp(x, M)


# ── Digital ───────────────────────────────────────────────────────

def brightness(x, severity=3):
    c = [25, 51, 76, 102, 127][severity - 1]
    return np.clip(np.array(x, dtype=np.float32) + c, 0, 255)


def stripe(x):
    out = np.array(x, dtype=np.float32)
    out[:, 11:17] = 255.0 - out[:, 11:17]
    return out


# ── Weather ───────────────────────────────────────────────────────

def fog(x, severity=3):
    c = [0.2, 0.35, 0.5, 0.65, 0.8][severity - 1]
    noise = cv2.GaussianBlur(np.random.uniform(0, 255, (28, 28)).astype(np.float32),
                             (25, 25), 8)
    noise = noise / noise.max() * 255
    return np.clip(np.array(x, dtype=np.float32) * (1 - c) + noise * c, 0, 255)


def spatter(x, severity=3):
    c = [0.1, 0.2, 0.3, 0.5, 0.7][severity - 1]
    mask = np.random.random((28, 28)).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (5, 5), 2)
    mask = (mask > (1 - c)).astype(np.float32)
    return np.clip(np.array(x, dtype=np.float32) * (1 - mask) + 63 * mask, 0, 255)


# ── Overlay ───────────────────────────────────────────────────────

def dotted_line(x):
    out = np.array(x, dtype=np.float32)
    r0 = np.random.randint(0, 28)
    r1 = np.random.randint(0, 28)
    img = np.zeros((28, 28), dtype=np.uint8)
    cv2.line(img, (0, r0), (27, r1), 255, 1)
    # make it dotted: blank every other 4-pixel segment
    for i in range(0, 28, 8):
        img[:, i:i+4] = 0
    return np.clip(out + img.astype(np.float32), 0, 255)


def zigzag(x):
    out = np.array(x, dtype=np.float32)
    img = np.zeros((28, 28), dtype=np.uint8)
    r = np.random.randint(5, 23)
    pts = []
    for col in range(0, 28, 4):
        offset = 4 if (col // 4) % 2 == 0 else -4
        pts.append([col, r + offset])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], False, 255, 1)
    return np.clip(out + img.astype(np.float32), 0, 255)


# ── Edge ──────────────────────────────────────────────────────────

def canny_edges(x):
    img = np.array(x, dtype=np.uint8)
    edges = cv2.Canny(img, 50, 150)
    return edges.astype(np.float32)
