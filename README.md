# EMNIST-C: A Corruption-Robustness Benchmark for Handwritten Letter Recognition

A corruption-robustness benchmark for the **EMNIST Letters** dataset, built by applying 15 image corruptions to the test set and measuring how a convolutional neural network holds up — then showing that corruption-augmented training largely closes the robustness gap.

Following the methodology of [MNIST-C (Mu & Gilmer, 2019)](https://arxiv.org/abs/1906.02337) and [CIFAR-10-C / ImageNet-C (Hendrycks & Dietterich, 2019)](https://arxiv.org/abs/1903.12261), this is the first full 15-corruption benchmark for handwritten *letters*, which are harder than digits due to visually similar pairs (C/G, I/L, M/W) and merged upper/lowercase variation.

> Course project — Machine Learning, Vrije Universiteit Amsterdam.
> Authors: Abdul Moeed Qadeer, Walid Ferjouchi, Amr Samaha.

## Key results

| Model | Clean accuracy | Mean accuracy across 15 corruptions |
|---|---|---|
| Baseline CNN | 91.4% | 65.5% |
| Corruption-augmented CNN (50% corruption prob.) | 92.9% | 88.7% |

Training on corrupted data recovers most of the lost robustness (+23 pp mean corrupted accuracy) **without** sacrificing clean accuracy. Per-corruption numbers are in the `*_results.json` files; figures are in the `*.png` files.

## The 15 corruptions

`shot_noise`, `impulse_noise`, `gaussian_blur`, `motion_blur`, `shear`, `scale`, `rotate`, `brightness`, `translate`, `stripe`, `fog`, `spatter`, `dotted_line`, `zigzag`, `canny_edges`.

All corruption functions are implemented from scratch with NumPy and OpenCV (`corruptions.py`); each maps a 28×28 float32 image in `[0, 255]` to the same shape and range. `glass_blur` is replaced with `gaussian_blur`, following CIFAR-10-C.

## Repository structure

```
corruptions.py          # 15 corruption functions (NumPy + OpenCV)
visualize.py            # Preview all corruptions on sample letters (sanity check)
build_emnist_c.py       # Apply all 15 corruptions to the EMNIST Letters test set -> emnist_c/*.npy
tune_baseline.py        # Optuna hyperparameter search for the baseline CNN
baseline.py             # Baseline CNN training / evaluation
train_augmented.py      # Train the corruption-augmented CNN
evaluate.py             # Evaluate a model across all corrupted test sets (supports --compare)
*.keras                 # Saved models (baseline, tuned baseline, augmented)
*_results.json          # Per-corruption accuracies
*.png                   # Figures (corruption samples, training history, comparisons)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate    # optional
pip install tensorflow tensorflow-datasets numpy opencv-python optuna matplotlib
```

EMNIST Letters is downloaded automatically via `tensorflow-datasets` on first run.

## Reproduce

```bash
# 1. (optional) Eyeball the corruptions before building anything
python visualize.py

# 2. Build the EMNIST-C corrupted test sets
python build_emnist_c.py

# 3. Find good baseline hyperparameters with Optuna
python tune_baseline.py

# 4. Train the corruption-augmented model
python train_augmented.py

# 5. Evaluate and compare baseline vs augmented across all corruptions
python evaluate.py --baseline tuned_baseline_cnn.keras --compare augmented_cnn.keras
```

## References

- Mu, N. & Gilmer, J. (2019). *MNIST-C: A Robustness Benchmark for Computer Vision.*
- Hendrycks, D. & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.*
- Cohen, G. et al. (2017). *EMNIST: an extension of MNIST to handwritten letters.*
- Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.*

## Authors

Abdul Moeed Qadeer · Walid Ferjouchi · Amr Samaha
