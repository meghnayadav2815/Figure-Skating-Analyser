"""
Action Classifier for Figure Skating Moves (v4 — Decomposed Dataset)
=====================================================================
Model 1 in the AI Skating Coach pipeline.

Trains a Bidirectional LSTM model on skeleton (pose) data from:
  - MMFS dataset: 2D pose estimation (63 action classes)
  - JSON mocap data: 3D motion capture (240 samples merged into jumps)
  - Decomposed combinations: Multi-jump sequences split into individual jumps
Total: 43 classes, 5948 samples (individual jumps + spins + steps)

Changes in v4:
  - Decomposed 21 combination classes into individual jumps
    * 2A+2T → 2Axel & 2Toeloop (separate samples)
    * 3Lz+3T → 3Lutz & 3Toeloop (separate samples)
    * Euler jumps skipped (no single-jump equivalent)
  - Simplified from 64 to 43 classes
  - Increased sample count by ~780 (5168 → 5948)
  - Massive boost to jump-specific classes:
    * 2Toeloop: 55 → 411 samples (+356)
    * 3Toeloop: 34 → 465 samples (+431)
    * 3Lutz: 207 → 468 samples (+261)
  - Imbalance ratio stable at 55.9x (35.6x after later oversampling)

Features from v3 (retained):
  - Combined MMFS + JSON datasets
  - Uniform temporal sampling
  - Class weights for handling imbalance
  - Data augmentation (mirror, noise, scaling)
  - Focal loss for imbalanced classification
  - 3-layer BiLSTM architecture

NOTE: Original 64-class dataset backed up at ~/skeleton/original_with_combos/
      Can train on combinations later using transfer learning from this model.
"""

import os
import pickle
import numpy as np
from collections import Counter

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Paths
DATA_DIR = os.path.expanduser('~/skeleton/')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, 'models')
PLOT_SAVE_DIR = os.path.join(SCRIPT_DIR, 'plots')

# Data parameters
MAX_SEQ_LENGTH = 500   # Pad/sample all sequences to this length (increased for longer context)
NUM_KEYPOINTS = 17     # Body joints per frame
NUM_COORDS = 3         # Coordinates per keypoint (x, y, confidence or z)
NUM_FEATURES = NUM_KEYPOINTS * NUM_COORDS  # 51 features per frame
NUM_CLASSES = 31       # Total action classes (filtered: >= 35 samples minimum)

# Model hyperparameters
DROPOUT_RATE = 0.35    # Dropout rate for regularization
LABEL_SMOOTHING = 0.1  # Smooth one-hot labels to help generalization
FOCAL_GAMMA = 2.0      # Focal loss gamma (down-weight easy examples)

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.003
EARLY_STOPPING_PATIENCE = 10  # Stop if val_loss doesn't improve for 10 epochs
VALIDATION_SPLIT = 0.2

# Data augmentation parameters
AUGMENTATION_ENABLED = False  # Disabled: using clean unaugmented data only

# Label-to-name mapping for confusion matrix
# Pruned to 31 classes (minimum 35 samples each)
LABEL_NAMES = {
    # Single jumps (0-16)
    0: "1Axel", 1: "1Flip", 2: "1Loop", 3: "1Lutz", 4: "2Axel",
    5: "2Flip", 6: "2Loop", 7: "2Lutz", 8: "2Salchow", 9: "2Toeloop",
    10: "3Axel", 11: "3Flip", 12: "3Loop", 13: "3Lutz", 14: "3Salchow", 15: "3Toeloop",
    16: "4Toeloop",
    # Spins and other elements (17-30)
    17: "ChCamelSp3", 18: "ChCamelSp4", 19: "ChComboSp2", 20: "ChComboSp3",
    21: "ChComboSp4", 22: "ChoreSeq1", 23: "ChSitSp4", 24: "FlySitSp3",
    25: "LaybackSp3", 26: "LaybackSp4", 27: "StepSeq1", 28: "StepSeq2",
    29: "StepSeq3", 30: "StepSeq4",
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
# 5. CLASS WEIGHTS
# ============================================================

print("\n  Skipping class weights (using raw data distribution)...")

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

# Add velocity features (temporal differences capture motion)
print("\n  Adding velocity features...")
train_real_mask = np.any(X_train != 0, axis=2, keepdims=True)
X_train_vel = np.diff(X_train, axis=1, prepend=X_train[:, :1, :])
X_train_vel *= train_real_mask  # zero out velocity at padding frames
X_train = np.concatenate([X_train, X_train_vel], axis=-1)

test_real_mask = np.any(X_test != 0, axis=2, keepdims=True)
X_test_vel = np.diff(X_test, axis=1, prepend=X_test[:, :1, :])
X_test_vel *= test_real_mask
X_test = np.concatenate([X_test, X_test_vel], axis=-1)

INPUT_FEATURES = X_train.shape[-1]  # 102 (51 position + 51 velocity)
print(f"  Features per frame: {INPUT_FEATURES} (position + velocity)")

# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Augmenting training data...")
print("=" * 60)


# Left-right joint pairs (COCO 17-keypoint format)
LR_PAIRS = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]


def mirror_skeleton(X, num_keypoints=17):
    """Mirror skeleton by negating x-coords and swapping left/right joints."""
    X_m = X.copy()
    feat_per_kp = 3
    # Process each feature block (position features, then velocity features)
    for base in range(0, X.shape[-1], num_keypoints * feat_per_kp):
        # Negate x coordinates
        for kp in range(num_keypoints):
            X_m[..., base + kp * feat_per_kp] *= -1
        # Swap left/right pairs
        for l, r in LR_PAIRS:
            li = slice(base + l * feat_per_kp, base + l * feat_per_kp + feat_per_kp)
            ri = slice(base + r * feat_per_kp, base + r * feat_per_kp + feat_per_kp)
            X_m[..., li], X_m[..., ri] = X_m[..., ri].copy(), X_m[..., li].copy()
    return X_m


def augment_data(X, y, noise_std=0.02, scale_range=(0.9, 1.1)):
    """
    Create augmented copies with mirror, noise, scaling.
    Returns 4x data: original + mirror + noisy + mirror_noisy.
    """
    aug_X, aug_y = [X], [y]
    mask = np.any(X != 0, axis=2, keepdims=True)

    # 1. Mirrored copy (left-right swap)
    X_mirror = mirror_skeleton(X)
    aug_X.append(X_mirror); aug_y.append(y)

    # 2. Noisy + randomly scaled
    X_ns = X.copy()
    X_ns += np.random.normal(0, noise_std, X_ns.shape).astype(np.float32) * mask
    X_ns *= np.random.uniform(*scale_range, (len(X), 1, 1)).astype(np.float32)
    X_ns *= mask
    aug_X.append(X_ns); aug_y.append(y)

    # 3. Mirrored + noisy + scaled
    X_mns = mirror_skeleton(X)
    m_mask = np.any(X_mns != 0, axis=2, keepdims=True)
    X_mns += np.random.normal(0, noise_std, X_mns.shape).astype(np.float32) * m_mask
    X_mns *= np.random.uniform(*scale_range, (len(X), 1, 1)).astype(np.float32)
    X_mns *= m_mask
    aug_X.append(X_mns); aug_y.append(y)

    X_out = np.concatenate(aug_X, axis=0)
    y_out = np.concatenate(aug_y, axis=0)
    idx = np.random.permutation(len(X_out))
    return X_out[idx], y_out[idx]


if AUGMENTATION_ENABLED:
    X_train_aug, y_train_aug = augment_data(
        X_train, y_train_onehot,
        noise_std=AUGMENTATION_NOISE_STD,
        scale_range=AUGMENTATION_SCALE_RANGE,
    )
    print(f"  Original training samples: {X_train.shape[0]}")
    print(f"  Augmented training samples: {X_train_aug.shape[0]} (4x with mirror)")
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

from tensorflow.keras.models import Model  # noqa: E402
from tensorflow.keras.layers import (  # noqa: E402
    LSTM, Bidirectional, Dense, Dropout, BatchNormalization,
    Conv1D, Input, Multiply, Softmax, Lambda, Add
)
from tensorflow.keras.optimizers import Adam  # noqa: E402
from tensorflow.keras.callbacks import (  # noqa: E402
    EarlyStopping, ModelCheckpoint, LearningRateScheduler
)

print(f"  TensorFlow version: {tf.__version__}")


def focal_loss(y_true, y_pred):
    """Focal loss with label smoothing — better for class-imbalanced data."""
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    y_smooth = y_true * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / NUM_CLASSES
    ce = -y_smooth * tf.math.log(y_pred)
    weight = tf.pow(1 - y_pred, FOCAL_GAMMA)
    return tf.reduce_sum(weight * ce, axis=-1)


input_features = X_train_aug.shape[-1]
inputs = Input(shape=(MAX_SEQ_LENGTH, input_features))

# --- Conv1D feature extraction (local temporal patterns) ---
# Kernel size reduced from 5 to 3, no max pooling
x = Conv1D(128, 3, padding='same', activation='relu')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Conv1D(256, 3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Conv1D(256, 3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

# --- First attention layer (after convolution, before LSTM) ---
attn1 = Dense(1)(x)
attn1_weights = Softmax(axis=1)(attn1)
attn1_output = Multiply()([x, attn1_weights])

# --- BiLSTM layer ---
lstm_out = Bidirectional(LSTM(160, return_sequences=True))(attn1_output)
lstm_out = BatchNormalization()(lstm_out)
lstm_out = Dropout(DROPOUT_RATE)(lstm_out)

# --- Second attention layer with skip connection from first attention ---
attn2 = Dense(1)(lstm_out)
attn2_weights = Softmax(axis=1)(attn2)
attn2_output = Multiply()([lstm_out, attn2_weights])

# Skip connection: add first attention output to second attention output
skip_connection = Add()([attn1_output, attn2_output])

# Global pooling
x = Lambda(lambda t: tf.reduce_sum(t, axis=1))(skip_connection)

# Classification head with 2 dense layers
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)  # Second dense layer with 128 units
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)

optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss=focal_loss,
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

def cosine_lr_schedule(epoch, lr):
    """Cosine annealing with 1-epoch warmup for fast convergence."""
    warmup_epochs = 1
    if epoch < warmup_epochs:
        return LEARNING_RATE * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(EPOCHS - warmup_epochs, 1)
    return LEARNING_RATE * 0.5 * (1 + np.cos(np.pi * progress))


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
    LearningRateScheduler(cosine_lr_schedule, verbose=1)
]

print("  Callbacks configured:")
print(f"    - EarlyStopping (patience={EARLY_STOPPING_PATIENCE}, monitor=val_loss)")
print("    - ModelCheckpoint (save best by val_accuracy)")
print(f"    - Cosine LR schedule (warmup 1 epoch, max_lr={LEARNING_RATE})")

# ============================================================
# 10. TRAIN THE MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Training the model...")
print("=" * 60)
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Training samples: {X_train_aug.shape[0]} (augmented)")
val_samples = int(X_train_aug.shape[0] * VALIDATION_SPLIT)
print(f"  Validation samples: {val_samples} (split={VALIDATION_SPLIT})")
print()

history = model.fit(
    X_train_aug, y_train_aug,
    validation_split=VALIDATION_SPLIT,
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
