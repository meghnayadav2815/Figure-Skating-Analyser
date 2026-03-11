"""
Action Classifier for Figure Skating Moves (v4 — Decomposed Dataset)
=====================================================================
Model 1 in the AI Skating Coach pipeline.

Trains a Bidirectional LSTM model on skeleton (pose) data from:
  - MMFS dataset: 2D pose estimation (63 action classes)
  - JSON mocap data: 3D motion capture (240 samples merged into jumps)
  - Decomposed combinations: Multi-jump sequences split into individual jumps
Total: 43 classes, 5948 samples (individual jumps + spins + steps)

Changes in v5:
  - Added 22 biomechanical skating features per frame (Stage A/B/C)
    Stage A — Approach direction (forward vs backward):
      * Torso facing angle (sin/cos), shoulder/hip line angles (sin/cos)
      * Movement direction (sin/cos), forward/backward indicator
    Stage B — Edge detection (inside vs outside):
      * Ankle lateral offset from hip center (L/R)
      * Knee bend angle via hip-knee-ankle cosine (L/R)
      * Ankle height difference (toe-pick proxy)
    Stage C — Pre-jump rotation & takeoff:
      * Shoulder-hip twist, shoulder/hip angular velocity
      * Twist rate (counter-rotation detector for Lutz vs Flip)
      * Vertical CoM velocity (takeoff/landing), body lean
      * Arm spread ratio, movement speed
  - Feature count: 51 skeleton + 22 skating = 73 per frame
  - With velocity: 146 features per frame (73 position + 73 velocity)

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
import re
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
MAX_SEQ_LENGTH = 500   # Pad/sample all sequences to this length
NUM_KEYPOINTS = 17     # Body joints per frame
NUM_COORDS = 3         # Coordinates per keypoint (x, y, confidence or z)
NUM_FEATURES = NUM_KEYPOINTS * NUM_COORDS  # 51 features per frame
NUM_CLASSES = 31       # Total action classes (filtered: >= 35 samples minimum)

# Model hyperparameters
DROPOUT_RATE = 0.3     # Dropout rate for regularization
LABEL_SMOOTHING = 0.1  # Smooth one-hot labels to help generalization
FOCAL_GAMMA = 2.0      # Focus on hard-to-classify examples (jumps), downweight easy ones

# Training hyperparameters
BATCH_SIZE = 32        # Smaller batches = more gradient updates per epoch
EPOCHS = 10  # Reduced for robust quick testing
LEARNING_RATE = 0.001  # With cosine decay

# Data filtering
EARLY_STOPPING_PATIENCE = 15  # Patient early stopping
VALIDATION_SPLIT = 0.2

# Data augmentation parameters
AUGMENTATION_ENABLED = True   # 4x data via mirror + noise + scaling
AUGMENTATION_NOISE_STD = 0.01
AUGMENTATION_SCALE_RANGE = (0.95, 1.05)

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

print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)
try:
    with open(os.path.join(DATA_DIR, 'train_data.pkl'), 'rb') as f:
        train_data_raw = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'train_label.pkl'), 'rb') as f:
        train_labels_raw = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'test_data.pkl'), 'rb') as f:
        test_data_raw = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'test_label.pkl'), 'rb') as f:
        test_labels_raw = pickle.load(f)
except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    raise

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


def extract_skating_features(X, num_keypoints=17):
    """
    Extract biomechanically-meaningful features for jump classification.

    Computes 22 features per frame organized into three discrimination stages:

    Stage A — Approach direction (forward vs backward):
      Axel is the ONLY jump taken going forward. All others are backward.
      Features: torso facing, shoulder/hip line angles, movement direction,
      and a forward/backward cosine indicator.

    Stage B — Edge detection (inside vs outside):
      Salchow/Flip = inside edge. Loop/Lutz/Toeloop = outside edge.
      Features: ankle lateral offset from hip center, knee bend angles,
      ankle height difference (toe-pick proxy).

    Stage C — Pre-jump rotation & takeoff characteristics:
      Lutz has counter-rotation (shoulder vs hip twist opposite to jump).
      Flip has natural rotation. Toe-pick jumps show different vertical
      acceleration at takeoff.
      Features: shoulder-hip twist, angular velocities, twist rate,
      vertical CoM velocity, body lean, arm spread, movement speed.

    Input:  X of shape (N, T, 51) — 17 COCO keypoints × 3 (x, y, conf/z)
    Output: features of shape (N, T, 22)

    COCO keypoint indices:
      0:Nose  1:L_Eye  2:R_Eye  3:L_Ear  4:R_Ear
      5:L_Shoulder  6:R_Shoulder  7:L_Elbow  8:R_Elbow
      9:L_Wrist  10:R_Wrist  11:L_Hip  12:R_Hip
      13:L_Knee  14:R_Knee  15:L_Ankle  16:R_Ankle
    """
    N, T, F = X.shape
    feat_per_kp = 3  # x, y, confidence/z

    def get_xy(kp_idx):
        """Extract x,y coordinates for a keypoint → (N, T, 2)."""
        return X[:, :, kp_idx * feat_per_kp : kp_idx * feat_per_kp + 2]

    # --- Key landmarks ---
    nose        = get_xy(0)
    l_shoulder  = get_xy(5);   r_shoulder = get_xy(6)
    l_hip       = get_xy(11);  r_hip      = get_xy(12)
    l_knee      = get_xy(13);  r_knee     = get_xy(14)
    l_ankle     = get_xy(15);  r_ankle    = get_xy(16)
    l_wrist     = get_xy(9);   r_wrist    = get_xy(10)

    shoulder_mid = (l_shoulder + r_shoulder) / 2.0  # (N, T, 2)
    hip_mid      = (l_hip + r_hip) / 2.0

    features = []

    # ================================================================
    # STAGE A: Approach direction (forward vs backward)
    # ================================================================

    # 1-2. Torso facing angle (hip→shoulder vector)
    torso_vec = shoulder_mid - hip_mid
    torso_angle = np.arctan2(torso_vec[:, :, 1], torso_vec[:, :, 0])  # (N, T)
    features.append(np.sin(torso_angle))  # [0] sin/cos avoids angle wrapping
    features.append(np.cos(torso_angle))  # [1]

    # 3-4. Shoulder line angle (rotation around vertical axis)
    shoulder_vec = r_shoulder - l_shoulder
    shoulder_angle = np.arctan2(shoulder_vec[:, :, 1], shoulder_vec[:, :, 0])
    features.append(np.sin(shoulder_angle))  # [2]
    features.append(np.cos(shoulder_angle))  # [3]

    # 5-6. Hip line angle
    hip_vec = r_hip - l_hip
    hip_angle = np.arctan2(hip_vec[:, :, 1], hip_vec[:, :, 0])
    features.append(np.sin(hip_angle))  # [4]
    features.append(np.cos(hip_angle))  # [5]

    # 7-8. Movement direction (velocity of hip midpoint)
    hip_vel = np.diff(hip_mid, axis=1, prepend=hip_mid[:, :1, :])
    move_angle = np.arctan2(hip_vel[:, :, 1], hip_vel[:, :, 0])
    features.append(np.sin(move_angle))  # [6]
    features.append(np.cos(move_angle))  # [7]

    # 9. Forward/backward indicator: cos(movement_dir − facing_dir)
    #    +1 = skating forward (Axel), −1 = skating backward (all others)
    forward_backward = np.cos(move_angle - torso_angle)
    features.append(forward_backward)  # [8]

    # ================================================================
    # STAGE B: Edge detection (inside vs outside)
    # ================================================================

    # 10-11. Ankle lateral offset from hip center
    #   Positive = foot outside body center, Negative = inside
    #   Inside edge (Salchow, Flip) vs outside edge (Loop, Lutz, Toeloop)
    l_ankle_offset = l_ankle[:, :, 0] - hip_mid[:, :, 0]
    r_ankle_offset = r_ankle[:, :, 0] - hip_mid[:, :, 0]
    features.append(l_ankle_offset)  # [9]
    features.append(r_ankle_offset)  # [10]

    # 12-13. Knee bend angle (cosine of hip-knee-ankle angle)
    #   Deep knee bend correlates with deeper edge and jump preparation
    def joint_angle_cos(a, b, c):
        """Cosine of angle at joint b formed by segments a→b and c→b."""
        ba = a - b  # (N, T, 2)
        bc = c - b
        dot = ba[:, :, 0] * bc[:, :, 0] + ba[:, :, 1] * bc[:, :, 1]
        norm = (np.sqrt(ba[:, :, 0]**2 + ba[:, :, 1]**2 + 1e-8) *
                np.sqrt(bc[:, :, 0]**2 + bc[:, :, 1]**2 + 1e-8))
        return np.clip(dot / norm, -1.0, 1.0)

    features.append(joint_angle_cos(l_hip, l_knee, l_ankle))   # [11]
    features.append(joint_angle_cos(r_hip, r_knee, r_ankle))   # [12]

    # 14. Ankle height difference (R − L)
    #   Toe-pick jumps (Flip, Lutz, Toeloop): one foot plants the toe
    #   while the other leg pushes — large difference at takeoff
    ankle_height_diff = r_ankle[:, :, 1] - l_ankle[:, :, 1]
    features.append(ankle_height_diff)  # [13]

    # ================================================================
    # STAGE C: Pre-jump rotation & takeoff
    # ================================================================

    # 15. Shoulder-hip twist (signed angular difference)
    #   Lutz requires counter-rotation — twist opposes jump direction
    twist = np.sin(shoulder_angle - hip_angle)
    features.append(twist)  # [14]

    # 16. Angular velocity of shoulders (rotation speed, rad/frame)
    shoulder_ang_vel = np.diff(shoulder_angle, axis=1, prepend=shoulder_angle[:, :1])
    shoulder_ang_vel = np.arctan2(np.sin(shoulder_ang_vel),
                                  np.cos(shoulder_ang_vel))  # wrap to [-π, π]
    features.append(shoulder_ang_vel)  # [15]

    # 17. Angular velocity of hips
    hip_ang_vel = np.diff(hip_angle, axis=1, prepend=hip_angle[:, :1])
    hip_ang_vel = np.arctan2(np.sin(hip_ang_vel), np.cos(hip_ang_vel))
    features.append(hip_ang_vel)  # [16]

    # 18. Twist rate (shoulder_vel − hip_vel)
    #   Large positive = upper body counter-rotating relative to hips (Lutz signature)
    #   Near zero = natural rotation (Flip)
    twist_rate = shoulder_ang_vel - hip_ang_vel
    features.append(twist_rate)  # [17]

    # 19. Vertical CoM velocity (approximate)
    #   Sharp upward = takeoff, sharp downward = landing
    com_y = (shoulder_mid[:, :, 1] + hip_mid[:, :, 1]) / 2.0
    com_y_vel = np.diff(com_y, axis=1, prepend=com_y[:, :1])
    features.append(com_y_vel)  # [18]

    # 20. Body lean (lateral displacement of hips relative to shoulders)
    lean = hip_mid[:, :, 0] - shoulder_mid[:, :, 0]
    features.append(lean)  # [19]

    # 21. Arm spread ratio (wrist distance / shoulder width)
    #   Arms pulled in during rotation → low ratio = mid-air spin
    #   Arms extended → high ratio = entry/exit
    wrist_dist = np.sqrt(np.sum((r_wrist - l_wrist)**2, axis=-1))
    shoulder_dist = np.sqrt(np.sum((r_shoulder - l_shoulder)**2, axis=-1))
    arm_spread = wrist_dist / (shoulder_dist + 1e-8)
    features.append(arm_spread)  # [20]

    # 22. Movement speed (hip velocity magnitude)
    #   Speed profile differs: Lutz has long glide, Loop has tight curve
    move_speed = np.sqrt(hip_vel[:, :, 0]**2 + hip_vel[:, :, 1]**2)
    features.append(move_speed)  # [21]

    # --- Stack and mask padding ---
    stacked = np.stack(features, axis=-1).astype(np.float32)  # (N, T, 22)

    # Zero out features where original skeleton data was padding
    padding_mask = np.all(X == 0, axis=2, keepdims=True)  # (N, T, 1)
    stacked[np.broadcast_to(padding_mask, stacked.shape)] = 0.0

    return stacked


NUM_SKATING_FEATURES = 22  # Biomechanical features from extract_skating_features


# Preprocess training data
print("  Processing training data (uniform temporal sampling)...")
X_train = preprocess_sequences(train_data_raw, MAX_SEQ_LENGTH, NUM_FEATURES, uniform_sample=True)
if np.isnan(X_train).any():
    print("[WARNING] NaNs detected in X_train. Replacing with zeros.")
    X_train = np.nan_to_num(X_train)
print(f"    X_train shape: {X_train.shape}")

# Preprocess test data
print("  Processing test data...")
X_test = preprocess_sequences(test_data_raw, MAX_SEQ_LENGTH, NUM_FEATURES, uniform_sample=True)
print(f"    X_test shape: {X_test.shape}")

# ============================================================
# 3a. CLIP OUTLIER VALUES
# ============================================================
# The dataset mixes 2D pose (confidence 0-1) with 3D mocap (z up to millions).
# This creates features with mean=19M, std=584M that dominate everything.
# Fix: clip every feature to its 1st-99th percentile range from training data.

print("\n  Clipping outlier values (per-feature 1st-99th percentile)...")
train_nonzero = np.any(X_train != 0, axis=2)  # (N, T) mask of real frames
real_frames = X_train[train_nonzero]  # (num_real_frames, 51)

clip_low = np.percentile(real_frames, 1, axis=0)   # (51,)
clip_high = np.percentile(real_frames, 99, axis=0)  # (51,)

print(f"    Before clip - value range: [{real_frames.min():.1f}, {real_frames.max():.1f}]")

# Clip training data (only non-padding frames)
for i in range(X_train.shape[-1]):
    col = X_train[:, :, i]
    mask = train_nonzero  # only clip real frames, leave padding as 0
    col[mask] = np.clip(col[mask], clip_low[i], clip_high[i])

# Clip test data with training percentiles
test_nonzero = np.any(X_test != 0, axis=2)
for i in range(X_test.shape[-1]):
    col = X_test[:, :, i]
    mask = test_nonzero
    col[mask] = np.clip(col[mask], clip_low[i], clip_high[i])

real_frames_after = X_train[train_nonzero]
print(f"    After clip  - value range: [{real_frames_after.min():.1f}, {real_frames_after.max():.1f}]")
del real_frames, real_frames_after

# ============================================================
# 3b. SKATING-SPECIFIC FEATURE EXTRACTION
# ============================================================

print("\n  Extracting skating-specific biomechanical features...")
print("    Stage A: Approach direction (forward/backward) — 9 features")
print("    Stage B: Edge detection (inside/outside)        — 5 features")
print("    Stage C: Pre-jump rotation & takeoff            — 8 features")

X_train_skating = extract_skating_features(X_train)
X_test_skating  = extract_skating_features(X_test)

print(f"    Skating features per frame: {X_train_skating.shape[-1]}")

# Concatenate skeleton + skating features (normalize together)
X_train = np.concatenate([X_train, X_train_skating], axis=-1)
if np.isnan(X_train).any():
    print("[WARNING] NaNs detected in X_train after feature concat. Replacing with zeros.")
    X_train = np.nan_to_num(X_train)
X_test  = np.concatenate([X_test, X_test_skating], axis=-1)
print(f"    Combined features per frame: {X_train.shape[-1]} "
      f"({NUM_FEATURES} skeleton + {NUM_SKATING_FEATURES} skating)")

del X_train_skating, X_test_skating  # free memory

# ============================================================
# 4. LABEL ENCODING
# ============================================================

print("\n  Encoding labels...")

try:
    y_train = np.array(train_labels_raw, dtype=np.int32)
except Exception as e:
    print(f"[ERROR] Failed to convert train_labels_raw to np.array: {e}")
    raise
y_test = np.array(test_labels_raw, dtype=np.int32)

if not (y_train.min() >= 0 and y_train.max() < NUM_CLASSES):
    raise ValueError(f"Train labels out of range: [{y_train.min()}, {y_train.max()}]")
assert y_test.min() >= 0 and y_test.max() < NUM_CLASSES, \
    f"Test labels out of range: [{y_test.min()}, {y_test.max()}]"

import tensorflow as tf  # noqa: E402
from tensorflow.keras.utils import to_categorical  # noqa: E402

# ============================================================
# 5. CLASS WEIGHTS & ONE-HOT ENCODING
# ============================================================

print("\n  Converting labels to one-hot encoding...")
y_train_onehot = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_onehot = to_categorical(y_test, num_classes=NUM_CLASSES)
print(f"    y_train_onehot shape: {y_train_onehot.shape}")
print(f"    y_test_onehot shape: {y_test_onehot.shape}")

print("\n  Computing class weights for imbalanced data...")
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# Display class distribution and weights
train_class_counts = Counter(y_train)
print(f"    Number of classes: {len(train_class_counts)}")
print(f"    Class distribution (train):")
for class_id in sorted(train_class_counts.keys())[:5]:
    count = train_class_counts[class_id]
    weight = class_weight_dict[class_id]
    class_name = LABEL_NAMES.get(class_id, str(class_id))
    print(f"      {class_name:15s}: {count:4d} samples, weight={weight:.3f}")
print(f"      ... ({NUM_CLASSES} classes total)")

# ============================================================
# 6. DATA NORMALIZATION
# ============================================================

print("\n  Normalizing data (vectorized)...")

# Compute mean and std from training data only on non-padding frames
# Create a mask of non-padding frames: shape (num_samples, max_len)
nonzero_mask = np.any(X_train != 0, axis=2)  # True where frame is not padding

# Collect all non-padding frames
all_frames = X_train[nonzero_mask]  # shape (num_real_frames, 73)
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

INPUT_FEATURES = X_train.shape[-1]  # 146 (73 position + 73 velocity)
print(f"  Features per frame: {INPUT_FEATURES} "
      f"({NUM_FEATURES + NUM_SKATING_FEATURES} position + "
      f"{NUM_FEATURES + NUM_SKATING_FEATURES} velocity)")

# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Augmenting training data...")
print("=" * 60)


# Left-right joint pairs (COCO 17-keypoint format)
LR_PAIRS = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]


def mirror_skeleton(X, num_keypoints=17):
    """
    Mirror skeleton by negating x-coords and swapping left/right joints.
    Now handles 146 features: 51 skeleton + 22 skating + 51 skel_vel + 22 skating_vel.
    Mirrors skeleton portions and recomputes skating features.
    """
    X_m = X.copy()
    feat_per_kp = 3
    num_skel_feat = num_keypoints * feat_per_kp  # 51
    total_pos_feat = num_skel_feat + NUM_SKATING_FEATURES  # 73
    
    # --- Mirror position features (first 73) ---
    # 1. Mirror skeleton keypoints (0-50)
    for kp in range(num_keypoints):
        X_m[..., kp * feat_per_kp] *= -1  # negate x coordinate
    
    # Swap left/right pairs in skeleton
    for l, r in LR_PAIRS:
        li = slice(l * feat_per_kp, l * feat_per_kp + feat_per_kp)
        ri = slice(r * feat_per_kp, r * feat_per_kp + feat_per_kp)
        X_m[..., li], X_m[..., ri] = X_m[..., ri].copy(), X_m[..., li].copy()
    
    # 2. Recompute skating features (51-72) from mirrored skeleton
    skel_mirrored = X_m[..., :num_skel_feat]  # (N, T, 51)
    # Reshape to (N, T, 51) if needed for extract_skating_features
    # extract_skating_features expects (N, T, 51)
    skating_feat_mirrored = extract_skating_features(skel_mirrored)  # (N, T, 22)
    X_m[..., num_skel_feat:total_pos_feat] = skating_feat_mirrored
    
    # --- Mirror velocity features (last 73) ---
    # 3. Mirror skeleton velocity keypoints (73-123)
    vel_offset = total_pos_feat
    for kp in range(num_keypoints):
        X_m[..., vel_offset + kp * feat_per_kp] *= -1  # negate x velocity
    
    # Swap left/right pairs in skeleton velocities
    for l, r in LR_PAIRS:
        li = slice(vel_offset + l * feat_per_kp, vel_offset + l * feat_per_kp + feat_per_kp)
        ri = slice(vel_offset + r * feat_per_kp, vel_offset + r * feat_per_kp + feat_per_kp)
        X_m[..., li], X_m[..., ri] = X_m[..., ri].copy(), X_m[..., li].copy()
    
    # 4. Recompute skating velocity features (124-145) from mirrored skeleton velocities
    skel_vel_mirrored = X_m[..., vel_offset:vel_offset + num_skel_feat]  # (N, T, 51)
    skating_vel_mirrored = extract_skating_features(skel_vel_mirrored)  # (N, T, 22)
    X_m[..., vel_offset + num_skel_feat:] = skating_vel_mirrored
    
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
    Conv1D, Input, Multiply, Softmax, Lambda
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


from tensorflow.keras.layers import GlobalAveragePooling1D, Concatenate, Add as AddLayer  # noqa: E402
from tensorflow.keras import regularizers  # noqa: E402

input_features = X_train_aug.shape[-1]
inputs = Input(shape=(MAX_SEQ_LENGTH, input_features))

# --- Conv1D blocks with residual connections ---
def conv_block(x, filters, kernel_size, dropout_rate=0.15):
    """Convolutional block with batch norm and residual connection."""
    shortcut = Conv1D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
    x = Conv1D(filters, kernel_size, padding='same', activation='relu', 
               kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = AddLayer()([shortcut, x])
    return x

# Conv1D doesn't support masking, so we skip Masking layer.
# Padding is already zeroed out; the attention mechanism learns to ignore it.

x = Conv1D(128, 3, padding='same', activation='relu')(inputs)
x = BatchNormalization()(x)

x = conv_block(x, 192, 3)
x = conv_block(x, 256, 3)
x = conv_block(x, 384, 5)

# --- Deep BiLSTM with residual ---
lstm1 = Bidirectional(LSTM(256, return_sequences=True))(x)
lstm1 = BatchNormalization()(lstm1)
lstm1 = Dropout(DROPOUT_RATE)(lstm1)

lstm2 = Bidirectional(LSTM(256, return_sequences=True))(lstm1)
lstm2 = BatchNormalization()(lstm2)
lstm2 = Dropout(DROPOUT_RATE)(lstm2)

lstm3 = Bidirectional(LSTM(256, return_sequences=True))(lstm2)
lstm3 = BatchNormalization()(lstm3)
lstm3 = Dropout(DROPOUT_RATE)(lstm3)

# Residual connection between LSTM layers
lstm_out = AddLayer()([lstm2, lstm3])

# --- Multi-scale attention: both global and local ---
# Global attention
attn_global = Dense(1)(lstm_out)
attn_weights_global = Softmax(axis=1)(attn_global)
attn_output_global = Multiply()([lstm_out, attn_weights_global])

# Local attention (different projection)
attn_local = Dense(1, activation='tanh')(lstm_out)
attn_weights_local = Softmax(axis=1)(attn_local)
attn_output_local = Multiply()([lstm_out, attn_weights_local])

# Pooling: combine max, avg, and multi-scale attention
max_pool = Lambda(lambda t: tf.reduce_max(t, axis=1))(lstm_out)
avg_pool = GlobalAveragePooling1D()(lstm_out)
attn_pool_global = Lambda(lambda t: tf.reduce_sum(t, axis=1))(attn_output_global)
attn_pool_local = Lambda(lambda t: tf.reduce_sum(t, axis=1))(attn_output_local)
x = Concatenate()([max_pool, avg_pool, attn_pool_global, attn_pool_local])

# --- Deep classification head with residual ---
x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Dense(256, activation='relu')(x)
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
    """Cosine annealing — start at LEARNING_RATE, decay smoothly to 1e-5."""
    min_lr = 1e-5
    progress = epoch / max(1, EPOCHS - 1)
    return min_lr + 0.5 * (LEARNING_RATE - min_lr) * (1 + np.cos(np.pi * progress))


callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        mode='max',
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
print(f"  Model will train for {EPOCHS} epochs (robust quick test mode)")
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
    class_weight=class_weight_dict,
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
