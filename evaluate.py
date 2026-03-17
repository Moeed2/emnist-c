"""
evaluate.py — Evaluate baseline on all EMNIST-C corrupted test sets.
"""

import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
from tensorflow import keras

CORRUPTIONS = [
    'shot_noise', 'impulse_noise', 'gaussian_blur', 'motion_blur',
    'shear', 'scale', 'rotate', 'brightness', 'translate',
    'stripe', 'fog', 'spatter', 'dotted_line', 'zigzag', 'canny_edges'
]


def evaluate_model(model, data_dir='emnist_c'):
    labels = np.load(os.path.join(data_dir, 'labels.npy'))

    # Clean
    x_clean = np.load(os.path.join(data_dir, 'identity.npy'))
    clean_acc = np.mean(np.argmax(model.predict(x_clean, verbose=0), axis=1) == labels)

    # Per corruption
    results = {}
    for name in CORRUPTIONS:
        x = np.load(os.path.join(data_dir, f'{name}.npy'))
        acc = np.mean(np.argmax(model.predict(x, verbose=0), axis=1) == labels)
        results[name] = acc

    return clean_acc, results


def print_table(clean_acc, results, name="Model"):
    clean_err = 1 - clean_acc
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"  Clean accuracy: {clean_acc*100:.2f}%")
    print(f"{'=' * 55}")
    print(f"  {'Corruption':<18} {'Accuracy':>9} {'Error':>8}")
    print(f"  {'-'*38}")
    for c in CORRUPTIONS:
        print(f"  {c:<18} {results[c]*100:>8.2f}% {(1-results[c])*100:>7.2f}%")
    mean_acc = np.mean(list(results.values()))
    print(f"  {'-'*38}")
    print(f"  {'MEAN':<18} {mean_acc*100:>8.2f}% {(1-mean_acc)*100:>7.2f}%")
    print(f"\n  Error increase: {(1-mean_acc)/clean_err:.1f}x vs clean")


def plot_bars(clean_acc, results, title="Baseline", path="results.png"):
    accs = [results[c] * 100 for c in CORRUPTIONS]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(CORRUPTIONS, accs, color='steelblue')
    ax.axhline(clean_acc * 100, color='red', linestyle='--', label=f'Clean ({clean_acc*100:.1f}%)')
    ax.axhline(np.mean(accs), color='orange', linestyle=':', label=f'Mean ({np.mean(accs):.1f}%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', type=str, default=None, help='Path to second model')
    parser.add_argument('--baseline', type=str, default='baseline_cnn.keras')
    args = parser.parse_args()

    # Baseline
    print("Loading baseline...")
    baseline = keras.models.load_model(args.baseline)
    clean_b, results_b = evaluate_model(baseline)
    print_table(clean_b, results_b, "Baseline CNN")
    plot_bars(clean_b, results_b, "Baseline CNN — EMNIST-C", "baseline_results.png")

    # Save
    out = {'clean': float(clean_b)}
    out.update({k: float(v) for k, v in results_b.items()})
    with open('baseline_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved baseline_results.json")

    # Compare
    if args.compare:
        print(f"\nLoading {args.compare}...")
        compare = keras.models.load_model(args.compare)
        clean_c, results_c = evaluate_model(compare)
        print_table(clean_c, results_c, "Augmented CNN")
        plot_bars(clean_c, results_c, "Augmented CNN — EMNIST-C", "augmented_results.png")

        # Side by side
        x = np.arange(len(CORRUPTIONS))
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.bar(x - 0.17, [results_b[c]*100 for c in CORRUPTIONS], 0.34, label='Baseline', color='steelblue')
        ax.bar(x + 0.17, [results_c[c]*100 for c in CORRUPTIONS], 0.34, label='Augmented', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(CORRUPTIONS, rotation=45, ha='right')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Baseline vs Augmented')
        ax.set_ylim(0, 105)
        ax.legend()
        plt.tight_layout()
        plt.savefig('comparison.png', dpi=150)
        plt.show()
        print("Saved comparison.png")
