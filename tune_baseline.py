"""
tune_baseline.py — Hyperparameter tuning with Optuna for the EMNIST Letters CNN.

What this script does (simple explanation):
  - It tries many different settings for the CNN automatically
  - Each 'trial' is one attempt with different settings
  - After all trials, it picks the best settings and trains the final model
  - The final model is saved as 'tuned_baseline_cnn.keras'

How to run:
  pip install optuna tensorflow tensorflow-datasets
  python tune_baseline.py

How long does it take?
  - Each trial takes roughly 2-4 minutes
  - With 20 trials that is about 40-80 minutes total
  - You can reduce N_TRIALS to 10 for a faster run
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import optuna
import json

# ── Settings ────────────────────────────────────────────────────────────────

N_TRIALS   = 20   # How many different settings to try (more = better but slower)
EPOCHS_TRY = 5    # Epochs per trial (kept short so trials are fast)
EPOCHS_FINAL = 20 # Epochs for the final best model (same as baseline)

# ── Data loading (same as baseline.py) ──────────────────────────────────────

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.transpose(image, perm=[1, 0, 2])  # EMNIST is stored transposed
    label = label - 1                             # Labels 1-26 → 0-25
    return image, label


print("Loading data...")
train_ds_raw = tfds.load('emnist/letters', split='train', as_supervised=True)
test_ds_raw  = tfds.load('emnist/letters', split='test',  as_supervised=True)

# We cache the preprocessed data once so every trial can reuse it fast
train_ds = (train_ds_raw
            .map(preprocess)
            .cache()
            .shuffle(10000)
            .prefetch(tf.data.AUTOTUNE))

test_ds = (test_ds_raw
           .map(preprocess)
           .cache()
           .prefetch(tf.data.AUTOTUNE))

print("Data loaded.\n")

# ── Model builder ────────────────────────────────────────────────────────────
#
# Optuna calls this function many times, each time with different values
# for the settings it is searching.
#
# Settings being searched:
#   filters1    — how many filters in the first Conv layer  (16 / 32 / 64)
#   filters2    — how many filters in the second Conv layer (32 / 64 / 128)
#   dense_units — size of the Dense layer                   (64 / 128 / 256)
#   dropout     — fraction of neurons randomly turned off   (0.0 – 0.5)
#   learning_rate — how fast the model learns               (0.0001 – 0.01)
#   batch_size  — how many images per training step         (64 / 128 / 256)

def build_model(trial):
    filters1     = trial.suggest_categorical('filters1',     [16, 32, 64])
    filters2     = trial.suggest_categorical('filters2',     [32, 64, 128])
    dense_units  = trial.suggest_categorical('dense_units',  [64, 128, 256])
    dropout      = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # First convolutional block
        layers.Conv2D(filters1, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        # Second convolutional block
        layers.Conv2D(filters2, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),

        # Dense layer with optional dropout
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout),

        # Output: 26 classes (A-Z)
        layers.Dense(26, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── Objective function ───────────────────────────────────────────────────────
#
# Optuna calls this once per trial.
# It builds and trains the model, then returns the validation accuracy.
# Optuna tries to MAXIMISE this number.

def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    # Batch the data with the batch size this trial wants to try
    tr = train_ds.batch(batch_size)
    te = test_ds.batch(batch_size)

    model = build_model(trial)

    # Only train for a few epochs during search (keeps it fast)
    history = model.fit(
        tr,
        epochs=EPOCHS_TRY,
        validation_data=te,
        verbose=0,   # Silent — Optuna prints its own progress
    )

    val_acc = max(history.history['val_accuracy'])

    # Free memory before next trial
    del model
    tf.keras.backend.clear_session()

    return val_acc


# ── Run the search ───────────────────────────────────────────────────────────

print(f"Starting Optuna search: {N_TRIALS} trials, {EPOCHS_TRY} epochs each")
print("This will take a while. Go grab a coffee ☕\n")

study = optuna.create_study(
    direction='maximize',          # We want the highest accuracy
    study_name='emnist_cnn_tuning',
    sampler=optuna.samplers.TPESampler(seed=42),  # Reproducible results
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ── Print results ────────────────────────────────────────────────────────────

best = study.best_trial
print("\n" + "=" * 55)
print("  BEST SETTINGS FOUND")
print("=" * 55)
for key, value in best.params.items():
    print(f"  {key:<20} {value}")
print(f"\n  Best val accuracy (5 epochs): {best.value * 100:.2f}%")
print("=" * 55)

# ── Train final model with best settings ─────────────────────────────────────

print(f"\nTraining final model with best settings for {EPOCHS_FINAL} epochs...")

best_batch = best.params['batch_size']
tr_final = train_ds.batch(best_batch)
te_final = test_ds.batch(best_batch)

final_model = build_model(best)
history_final = final_model.fit(
    tr_final,
    epochs=EPOCHS_FINAL,
    validation_data=te_final,
    verbose=1,
)

loss, acc = final_model.evaluate(te_final, verbose=0)
print(f"\nFinal tuned model — Clean test accuracy: {acc * 100:.2f}%")

# Save the model
final_model.save('tuned_baseline_cnn.keras')
print("Saved to tuned_baseline_cnn.keras")

# Save best params + accuracy to JSON (useful for the report)
results = {
    'best_params': best.params,
    'best_val_acc_during_search': float(best.value),
    'final_clean_accuracy': float(acc),
}
with open('tuned_baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved tuned_baseline_results.json")

# ── Plot training history ─────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history_final.history['accuracy'],     label='Train')
ax1.plot(history_final.history['val_accuracy'], label='Val')
ax1.set_title('Tuned Model — Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history_final.history['loss'],     label='Train')
ax2.plot(history_final.history['val_loss'], label='Val')
ax2.set_title('Tuned Model — Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig('tuned_training_history.png', dpi=150)
plt.show()
print("Saved tuned_training_history.png")

# ── Plot all trials (which settings scored what) ──────────────────────────────

trial_numbers   = [t.number for t in study.trials]
trial_accuracies = [t.value * 100 for t in study.trials]

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(trial_numbers, trial_accuracies, color='steelblue')
ax.axhline(best.value * 100, color='red', linestyle='--',
           label=f'Best ({best.value*100:.2f}%)')
ax.set_xlabel('Trial number')
ax.set_ylabel('Val accuracy after 5 epochs (%)')
ax.set_title('Optuna — Accuracy per trial')
ax.legend()
plt.tight_layout()
plt.savefig('optuna_trials.png', dpi=150)
plt.show()
print("Saved optuna_trials.png")

print("\nAll done! Use tuned_baseline_cnn.keras as your improved baseline.")
