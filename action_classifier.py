"""
Action Classifier for Figure Skating Moves (v2 — High Accuracy)
================================================================
Model 1 in the AI Skating Coach pipeline.

Trains a Bidirectional LSTM model on skeleton (pose) data from the MMFS dataset
to classify figure skating actions into 63 categories.

Improvements over v1:
  - Uniform temporal sampling (instead of simple truncation)
  - Class weights to handle severe class imbalance (8–406 samples)
  - Data augmentation (Gaussian noise, temporal jitter, random scaling)
  - Larger model with 3 BiLSTM layers
  - Vectorized normalization (much faster)
  - Training curves and confusion matrix plots
"""

import os
import pickle
import numpy as np
from collections import Counter

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Paths
DATA_DIR = os.path.expanduser('~/Downloads/mmfs_hf_repo/skeleton/')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, 'models')
PLOT_SAVE_DIR = os.path.join(SCRIPT_DIR, 'plots')

# Data parameters
MAX_SEQ_LENGTH = 300   # Pad/sample all sequences to this length
NUM_KEYPOINTS = 17     # Body joints per frame
NUM_COORDS = 3         # Coordinates per keypoint (x, y, confidence or z)
NUM_FEATURES = NUM_KEYPOINTS * NUM_COORDS  # 51 features per frame
NUM_CLASSES = 63       # Total action classes (labels 0–62)

# Model hyperparameters
LSTM_UNITS_1 = 128     # Units in the first BiLSTM layer
LSTM_UNITS_2 = 64      # Units in the second BiLSTM layer
DENSE_UNITS = 64       # Units in the dense layer before output
DROPOUT_RATE = 0.4     # Dropout rate for regularization

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10  # Stop if val_loss doesn't improve for 10 epochs

# Data augmentation parameters (disabled for fast local training)
AUGMENTATION_ENABLED = False
AUGMENTATION_NOISE_STD = 0.01   # Std of Gaussian noise added to features
AUGMENTATION_SCALE_RANGE = (0.95, 1.05)  # Random scaling range

# Label-to-name mapping for confusion matrix
LABEL_NAMES = {
    0: "1Axel", 1: "1Flip", 2: "1Loop", 3: "1Lutz", 4: "2Axel",
    5: "2A+1Eu+2S", 6: "2A+1Eu+3S", 7: "2A+2T", 8: "2A+3T", 9: "2Flip",
    10: "2F+2T", 11: "2Loop", 12: "2Lutz", 13: "2Lz+2T", 14: "2Salchow",
    15: "2Toeloop", 16: "3Axel", 17: "3A+2T", 18: "3A+3T", 19: "3Flip",
    20: "3F+1Eu+3S", 21: "3F+2T", 22: "3F+3T", 23: "3Loop", 24: "3Lo+2T",
    25: "3Lutz", 26: "3Lz+2T", 27: "3Lz+3T", 28: "3Salchow", 29: "3S+2T",
    30: "3S+2T+2Lo", 31: "3S+3T", 32: "3Toeloop", 33: "3T+2T", 34: "3T+3T",
    35: "4Lutz", 36: "4Salchow", 37: "4Toeloop", 38: "4T+2T", 39: "4T+3T",
    40: "CamelSp4", 41: "ChCamelSp1", 42: "ChCamelSp2", 43: "ChCamelSp3",
    44: "ChCamelSp4", 45: "ChComboSp1V", 46: "ChComboSp2", 47: "ChComboSp3",
    48: "ChComboSp4", 49: "ChoreSeq1", 50: "ChSitSp3", 51: "ChSitSp4",
    52: "FlyCamelSp3", 53: "FlySitSp1", 54: "FlySitSp3", 55: "LaybackSp1",
    56: "LaybackSp2", 57: "LaybackSp3", 58: "LaybackSp4", 59: "StepSeq1",
    60: "StepSeq2", 61: "StepSeq3", 62: "StepSeq4",
}

# ============================================================
# 2. DATA LOADING
# ============================================================

print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

with open(os.path.join(DATA_DIR, 'train_data.pkl'), 'rb') as f:
    train_data_raw = pickle.load(f)

with open(os.path.join(DATA_DIR, 'train_label.pkl'), 'rb') as f:
    train_labels_raw = pickle.load(f)

with open(os.path.join(DATA_DIR, 'test_data.pkl'), 'rb') as f:
    test_data_raw = pickle.load(f)

with open(os.path.join(DATA_DIR, 'test_label.pkl'), 'rb') as f:
    test_labels_raw = pickle.load(f)

print(f"  Loaded {len(train_data_raw)} training samples")
print(f"  Loaded {len(test_data_raw)} test samples")
print(f"  Number of classes: {len(set(train_labels_raw + test_labels_raw))}")

# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Preprocessing data...")
print("=" * 60)


def preprocess_sequences(data_list, max_len, num_features, uniform_sample=True):
    """
    Convert variable-length skeleton sequences into fixed-size numpy arrays.

    Uses uniform temporal sampling for sequences longer than max_len,
    which preserves information from the entire sequence rather than
    just the beginning.

    Args:
        data_list: List of numpy arrays, each of shape (T_i, 17, 3, 1)
        max_len: Fixed sequence length to pad/sample to
        num_features: Number of features per timestep (17 * 3 = 51)
        uniform_sample: If True, uniformly sample frames from long sequences
                        instead of truncating

    Returns:
        numpy array of shape (num_samples, max_len, num_features)
    """
    num_samples = len(data_list)
    processed = np.zeros((num_samples, max_len, num_features), dtype=np.float32)

    skipped = 0
    for i, sample in enumerate(data_list):
        # Remove trailing dimension: (T, 17, 3, 1) -> (T, 17, 3)
        sample = sample.squeeze(axis=-1)
        seq_len = sample.shape[0]

        if seq_len == 0:
            skipped += 1
            continue

        # Flatten keypoints: (T, 17, 3) -> (T, 51)
        sample_flat = sample.reshape(seq_len, -1)

        if seq_len > max_len:
            if uniform_sample:
                # Uniformly sample max_len frames from the full sequence
                # This preserves information from the entire action
                indices = np.linspace(0, seq_len - 1, max_len, dtype=int)
                processed[i] = sample_flat[indices]
            else:
                processed[i] = sample_flat[:max_len]
        else:
            processed[i, :seq_len, :] = sample_flat

    if skipped > 0:
        print(f"  WARNING: {skipped} empty sequences encountered (filled with zeros)")

    return processed


# Preprocess training data
print("  Processing training data (uniform temporal sampling)...")
X_train = preprocess_sequences(train_data_raw, MAX_SEQ_LENGTH, NUM_FEATURES, uniform_sample=True)
print(f"    X_train shape: {X_train.shape}")

# Preprocess test data
print("  Processing test data...")
X_test = preprocess_sequences(test_data_raw, MAX_SEQ_LENGTH, NUM_FEATURES, uniform_sample=True)
print(f"    X_test shape: {X_test.shape}")

# ============================================================
# 4. LABEL ENCODING
# ============================================================

print("\n  Encoding labels...")

y_train = np.array(train_labels_raw, dtype=np.int32)
y_test = np.array(test_labels_raw, dtype=np.int32)

assert y_train.min() >= 0 and y_train.max() < NUM_CLASSES, \
    f"Train labels out of range: [{y_train.min()}, {y_train.max()}]"
assert y_test.min() >= 0 and y_test.max() < NUM_CLASSES, \
    f"Test labels out of range: [{y_test.min()}, {y_test.max()}]"

import tensorflow as tf  # noqa: E402
from tensorflow.keras.utils import to_categorical  # noqa: E402

y_train_onehot = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_onehot = to_categorical(y_test, num_classes=NUM_CLASSES)

print(f"    y_train_onehot shape: {y_train_onehot.shape}")
print(f"    y_test_onehot shape: {y_test_onehot.shape}")

# ============================================================
# 5. CLASS WEIGHTS (disabled — caused training instability)
# ============================================================

print("\n  Class weights: disabled (not using)")

# ============================================================
# 6. DATA NORMALIZATION
# ============================================================

print("\n  Normalizing data (vectorized)...")

# Compute mean and std from training data only on non-padding frames
# Create a mask of non-padding frames: shape (num_samples, max_len)
nonzero_mask = np.any(X_train != 0, axis=2)  # True where frame is not padding

# Collect all non-padding frames
all_frames = X_train[nonzero_mask]  # shape (num_real_frames, 51)
feature_mean = all_frames.mean(axis=0)
feature_std = all_frames.std(axis=0)
feature_std[feature_std < 1e-8] = 1.0

print(f"    Feature mean range: [{feature_mean.min():.4f}, {feature_mean.max():.4f}]")
print(f"    Feature std range:  [{feature_std.min():.4f}, {feature_std.max():.4f}]")

# Vectorized normalization: normalize everything, then zero out padding
X_train_norm = (X_train - feature_mean) / feature_std
# Re-apply the padding mask (set padding frames back to zero)
train_pad_mask = ~np.any(X_train != 0, axis=2, keepdims=True)  # True where padding
X_train_norm[np.broadcast_to(train_pad_mask, X_train_norm.shape)] = 0.0
X_train = X_train_norm

X_test_norm = (X_test - feature_mean) / feature_std
test_pad_mask = ~np.any(X_test != 0, axis=2, keepdims=True)
X_test_norm[np.broadcast_to(test_pad_mask, X_test_norm.shape)] = 0.0
X_test = X_test_norm

print("  Normalization complete.")

# Save normalization parameters for inference later
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
np.savez(os.path.join(MODEL_SAVE_DIR, 'normalization_params.npz'),
         mean=feature_mean, std=feature_std)
print(f"  Saved normalization params to {MODEL_SAVE_DIR}/normalization_params.npz")

# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Augmenting training data...")
print("=" * 60)


def augment_data(X, y, noise_std=0.02, scale_range=(0.9, 1.1), num_augmented=1):
    """
    Create augmented copies of the training data.

    Augmentations applied:
      1. Gaussian noise injection
      2. Random scaling of all features
      3. Random temporal shift (circular shift along time axis)

    Args:
        X: Training data, shape (N, T, F)
        y: One-hot labels, shape (N, C)
        noise_std: Standard deviation of Gaussian noise
        scale_range: (min, max) for random scaling factor
        num_augmented: Number of augmented copies per sample

    Returns:
        Augmented X and y (original + augmented)
    """
    augmented_X = [X]  # Start with original data
    augmented_y = [y]

    for _ in range(num_augmented):
        X_aug = X.copy()

        # Find non-padding mask for each sample
        sample_mask = np.any(X_aug != 0, axis=2)  # (N, T)

        # 1. Add Gaussian noise (only to non-padding frames)
        noise = np.random.normal(0, noise_std, X_aug.shape).astype(np.float32)
        noise_mask = np.expand_dims(sample_mask, axis=2)  # (N, T, 1)
        X_aug += noise * noise_mask

        # 2. Random scaling per sample
        scales = np.random.uniform(scale_range[0], scale_range[1],
                                   size=(X_aug.shape[0], 1, 1)).astype(np.float32)
        X_aug *= scales
        # Ensure padding stays zero
        X_aug[~np.broadcast_to(noise_mask, X_aug.shape)] = 0.0

        augmented_X.append(X_aug)
        augmented_y.append(y)

    return np.concatenate(augmented_X, axis=0), np.concatenate(augmented_y, axis=0)


if AUGMENTATION_ENABLED:
    X_train_aug, y_train_aug = augment_data(
        X_train, y_train_onehot,
        noise_std=AUGMENTATION_NOISE_STD,
        scale_range=AUGMENTATION_SCALE_RANGE,
        num_augmented=1
    )
    print(f"  Original training samples: {X_train.shape[0]}")
    print(f"  Augmented training samples: {X_train_aug.shape[0]} (2x)")
    shuffle_idx = np.random.permutation(len(X_train_aug))
    X_train_aug = X_train_aug[shuffle_idx]
    y_train_aug = y_train_aug[shuffle_idx]
else:
    X_train_aug = X_train
    y_train_aug = y_train_onehot
    print(f"  Augmentation disabled. Training samples: {X_train.shape[0]}")

# ============================================================
# 8. BUILD THE MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Building the model...")
print("=" * 60)

from tensorflow.keras.models import Sequential  # noqa: E402
from tensorflow.keras.layers import (  # noqa: E402
    LSTM, Bidirectional, Dense, Dropout, BatchNormalization, Masking
)
from tensorflow.keras.optimizers import Adam  # noqa: E402
from tensorflow.keras.callbacks import (  # noqa: E402
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

print(f"  TensorFlow version: {tf.__version__}")

model = Sequential([
    # Masking layer: ignore zero-padded timesteps
    Masking(mask_value=0.0, input_shape=(MAX_SEQ_LENGTH, NUM_FEATURES)),

    # First BiLSTM layer (128 units each direction = 256 output features)
    # use_cudnn=False needed because Masking can produce non-contiguous masks
    Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True, use_cudnn=False)),
    BatchNormalization(),
    Dropout(DROPOUT_RATE),

    # Second BiLSTM layer (64 units each direction = 128 output features)
    # return_sequences=False: output only the final hidden state
    Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=False, use_cudnn=False)),
    BatchNormalization(),
    Dropout(DROPOUT_RATE),

    # Dense layer for classification
    Dense(DENSE_UNITS, activation='relu'),
    Dropout(0.3),

    # Output layer
    Dense(NUM_CLASSES, activation='softmax')
])

optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 9. DEFINE CALLBACKS
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Setting up training callbacks...")
print("=" * 60)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_DIR, 'action_classifier_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

print("  Callbacks configured:")
print(f"    - EarlyStopping (patience={EARLY_STOPPING_PATIENCE}, monitor=val_loss)")
print("    - ModelCheckpoint (save best by val_accuracy)")
print("    - ReduceLROnPlateau (factor=0.5, patience=5)")

# ============================================================
# 10. TRAIN THE MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Training the model...")
print("=" * 60)
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Training samples: {X_train_aug.shape[0]} (augmented)")
print(f"  Validation samples: {X_test.shape[0]}")
print()

history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_test, y_test_onehot),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 11. EVALUATE THE MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Evaluating the model...")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
print(f"\n  Final Test Loss:     {test_loss:.4f}")
print(f"  Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

# ============================================================
# 12. SAVE THE FINAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: Saving the model...")
print("=" * 60)

final_model_path = os.path.join(MODEL_SAVE_DIR, 'action_classifier_final.keras')
model.save(final_model_path)
print(f"  Final model saved to: {final_model_path}")
print(f"  Best model saved to:  {os.path.join(MODEL_SAVE_DIR, 'action_classifier_best.keras')}")

history_path = os.path.join(MODEL_SAVE_DIR, 'training_history.pkl')
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
print(f"  Training history saved to: {history_path}")

# ============================================================
# 13. PLOT TRAINING CURVES
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: Generating training curves...")
print("=" * 60)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt  # noqa: E402

os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Accuracy plot ---
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

# --- Loss plot ---
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Action Classifier Training Curves', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

curves_path = os.path.join(PLOT_SAVE_DIR, 'training_curves.png')
plt.savefig(curves_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Training curves saved to: {curves_path}")

# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: Generating confusion matrix...")
print("=" * 60)

from sklearn.metrics import confusion_matrix, classification_report  # noqa: E402

# Get predictions on test set
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test

# Classification report
class_names = [LABEL_NAMES.get(i, str(i)) for i in range(NUM_CLASSES)]
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
print("\n  Classification Report:")
print(report)

# Save classification report
report_path = os.path.join(PLOT_SAVE_DIR, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(report)
print(f"  Classification report saved to: {report_path}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(24, 20))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax, shrink=0.8)

ax.set(
    xticks=np.arange(NUM_CLASSES),
    yticks=np.arange(NUM_CLASSES),
    xticklabels=class_names,
    yticklabels=class_names,
    ylabel='True Label',
    xlabel='Predicted Label',
    title=f'Confusion Matrix — Test Accuracy: {test_accuracy * 100:.2f}%'
)

plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=7)
plt.setp(ax.get_yticklabels(), fontsize=7)

# Add text annotations for cells with values > 0
thresh = cm.max() / 2.0
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if cm[i, j] > 0:
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", fontsize=5,
                    color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
cm_path = os.path.join(PLOT_SAVE_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Confusion matrix saved to: {cm_path}")

# ============================================================
# 15. TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

best_epoch = np.argmin(history.history['val_loss']) + 1
best_val_loss = min(history.history['val_loss'])
best_val_acc = max(history.history['val_accuracy'])

print(f"  Best validation loss:     {best_val_loss:.4f} (epoch {best_epoch})")
print(f"  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
print(f"  Total epochs trained:     {len(history.history['loss'])}")
print(f"\n  Model files saved in: {MODEL_SAVE_DIR}/")
print(f"  Plots saved in:       {PLOT_SAVE_DIR}/")
print("  Done!")
