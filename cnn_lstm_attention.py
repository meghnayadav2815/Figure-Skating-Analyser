"""
Action Classifier for Figure Skating Moves (v4 — Decomposed Dataset) with Multi-Head Attention
=====================================================================
This script is a copy of cnn + LSTM.py, modified to add a MultiHeadAttention layer after the BiLSTM stack.
"""

import os
import pickle
import re
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from tensorflow.keras.utils import to_categorical
from collections import Counter

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Paths
DATA_DIR = '/root/skeleton/'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, 'models')
PLOT_SAVE_DIR = os.path.join(SCRIPT_DIR, 'plots')

# Data parameters
MAX_SEQ_LENGTH = 500   # Pad/sample all sequences to this length
NUM_KEYPOINTS = 17     # Body joints per frame
NUM_COORDS = 3         # Coordinates per keypoint (x, y, confidence or z)
NUM_FEATURES = NUM_KEYPOINTS * NUM_COORDS  # 51 features per frame
NUM_CLASSES = None     # Will be loaded from label_mapping.json after data loading

# Model hyperparameters
DROP_OUT_RATE = 0.15   # Lower dropout to allow more learning signal through
BATCH_NORM_AFTER_DENSE = True
LABEL_SMOOTHING = 0.05  # Reduced label smoothing for sharper learning
FOCAL_GAMMA = 1.5      # Softer focal loss focusing

# Training hyperparameters
BACTH_SIZE = 16        # Smaller batch = more gradient updates per epoch
EPOCHS = 20  # Increased from 10 to allow better convergence
LEARNING_RATE = 0.001  # Moderate learning rate

# Data filtering
EARLY_STOPPING_PATIENCE = 15  # Patient stopping to allow learning
VALIDATION_SPLIT = 0.15  # Standard validation split

# Data augmentation parameters
AUGMENTATION_ENABLED = True   # 4x data via mirror + noise + scaling
AUGMENTATION_NOISE_STD = 0.02  # Moderate-high noise
AUGMENTATION_SCALE_RANGE = (0.90, 1.10)  # Moderate-wide scaling

# Label-to-name mapping for confusion matrix
# Will be loaded dynamically from label_mapping.json after data loading
LABEL_NAMES = {}

# ...[DATA LOADING, PREPROCESSING, AUGMENTATION, ETC. UNCHANGED]...
# ===================== Robust Data Loading =====================
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

# ===================== Load label mapping dynamically =====================
import json
try:
	with open(os.path.join(DATA_DIR, 'label_mapping.json'), 'r') as f:
		label_mapping = json.load(f)
	LABEL_NAMES = {int(k): v for k, v in label_mapping.items()}
	NUM_CLASSES = len(LABEL_NAMES)
	print(f"✅ Loaded {NUM_CLASSES} classes from label_mapping.json")
	print(f"   Classes: {list(LABEL_NAMES.values())}")
except Exception as e:
	print(f"[WARNING] Could not load label_mapping.json: {e}")
	print(f"          Using default NUM_CLASSES = 63 (ungrouped dataset)")
	NUM_CLASSES = 63
	LABEL_NAMES = {i: f"class_{i}" for i in range(NUM_CLASSES)}

# ===================== Preprocessing NaN Checks =====================

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

# --- Use a true validation set (20% of original data, not augmented) ---
from sklearn.model_selection import train_test_split
X_all = preprocess_sequences(train_data_raw, MAX_SEQ_LENGTH, NUM_FEATURES, uniform_sample=True)
if np.isnan(X_all).any():
	print("[WARNING] NaNs detected in X_all. Replacing with zeros.")
	X_all = np.nan_to_num(X_all)
try:
	y_all = np.array(train_labels_raw, dtype=np.int32)
except Exception as e:
	print(f"[ERROR] Failed to convert train_labels_raw to np.array: {e}")
	raise
if not (y_all.min() >= 0 and y_all.max() < NUM_CLASSES):
	raise ValueError(f"Train labels out of range: [{y_all.min()}, {y_all.max()}]")

X_train, X_val, y_train, y_val = train_test_split(
	X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

print(f"  Training samples: {X_train.shape[0]}")
print(f"  Validation samples: {X_val.shape[0]}")

y_train_onehot = to_categorical(y_train, num_classes=NUM_CLASSES)
y_val_onehot = to_categorical(y_val, num_classes=NUM_CLASSES)
print(f"    y_train_onehot shape: {y_train_onehot.shape}")
print(f"    y_val_onehot shape: {y_val_onehot.shape}")

# --- Compute class weights for imbalanced classes ---
class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# --- Data Augmentation Utilities ---
def mirror_skeleton(X, num_keypoints=17):
	X_m = X.copy()
	feat_per_kp = 3
	num_skel_feat = num_keypoints * feat_per_kp  # 51
	# 1. Mirror skeleton keypoints (0-50)
	for kp in range(num_keypoints):
		X_m[..., kp * feat_per_kp] *= -1
	# No left/right swap for minimal version
	return X_m

def augment_data(X, y, noise_std=0.02, scale_range=(0.90, 1.10)):
	"""
	4x augmentation: mirror + variations.
	Keep simple to avoid confusing the model.
	"""
	aug_X, aug_y = [X], [y]  # Original
	mask = np.any(X != 0, axis=2, keepdims=True)
	
	# 1. Mirrored
	X_mirror = mirror_skeleton(X)
	aug_X.append(X_mirror); aug_y.append(y)
	
	# 2. Noisy + scaled
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

# --- Apply Augmentation or Fallback ---

if AUGMENTATION_ENABLED:
	X_train_aug, y_train_aug = augment_data(
		X_train, y_train_onehot,
		noise_std=AUGMENTATION_NOISE_STD,
		scale_range=AUGMENTATION_SCALE_RANGE,
	)
	print(f"  Original training samples: {X_train.shape[0]}")
	print(f"  Augmented training samples: {X_train_aug.shape[0]} (4x with mirror + noise + scale)")
else:
	X_train_aug = X_train
	y_train_aug = y_train_onehot
	print(f"  Augmentation disabled. Training samples: {X_train.shape[0]}")

# If you concatenate features, check again:
# X_train = np.concatenate([X_train, X_train_skating], axis=-1)
# if np.isnan(X_train).any():
#     print("[WARNING] NaNs detected in X_train after feature concat. Replacing with zeros.")
#     X_train = np.nan_to_num(X_train)

try:
	y_train = np.array(train_labels_raw, dtype=np.int32)
except Exception as e:
	print(f"[ERROR] Failed to convert train_labels_raw to np.array: {e}")
	raise
if not (y_train.min() >= 0 and y_train.max() < NUM_CLASSES):
	raise ValueError(f"Train labels out of range: [{y_train.min()}, {y_train.max()}]")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
	Dense, Dropout, BatchNormalization,
	Conv1D, Input, Multiply, Softmax, Lambda, GlobalAveragePooling1D, Concatenate, Add as AddLayer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras import regularizers

# --- MultiHeadAttention import ---
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

# ...[focal_loss and other utility functions unchanged]...

def focal_loss(y_true, y_pred):
	"""Focal loss with label smoothing — better for class-imbalanced data."""
	y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
	y_smooth = y_true * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / NUM_CLASSES
	ce = -y_smooth * tf.math.log(y_pred)
	weight = tf.pow(1 - y_pred, FOCAL_GAMMA)
	return tf.reduce_sum(weight * ce, axis=-1)


# --- Model Input Features ---
input_features = X_train_aug.shape[-1]
inputs = Input(shape=(MAX_SEQ_LENGTH, input_features))

# --- Conv1D blocks with residual connections ---
def conv_block(x, filters, kernel_size, dropout_rate=0.15):
	shortcut = Conv1D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
	x = Conv1D(filters, kernel_size, padding='same', activation='relu', 
			   kernel_regularizer=regularizers.l2(1e-4))(x)
	x = BatchNormalization()(x)
	x = Dropout(dropout_rate)(x)
	x = AddLayer()([shortcut, x])
	return x


# --- Convolutional feature extractor (wider backbone) ---
x = Conv1D(128, 3, padding='same', activation='relu')(inputs)
x = BatchNormalization()(x)
x = conv_block(x, 192, 3)
x = conv_block(x, 256, 3)
x = conv_block(x, 384, 5)

# --- Attention stack (larger model dimension + more heads) ---
attn_conv = MultiHeadAttention(
	num_heads=8,
	key_dim=64,
	output_shape=384,
	dropout=0.1,
	name="attn_after_conv"
)(x, x)
attn_conv = AddLayer(name="resid_attn_conv")([x, attn_conv])
attn_conv = LayerNormalization()(attn_conv)
attn_conv = Dropout(0.2)(attn_conv)



# --- Additional Attention block ---
attn2 = MultiHeadAttention(
	num_heads=8,
	key_dim=64,
	output_shape=384,
	dropout=0.1,
	name="attn_after_conv2"
)(attn_conv, attn_conv)
attn2 = AddLayer(name="resid_attn_conv2")([attn_conv, attn2])
attn2 = LayerNormalization()(attn2)
attn2 = Dropout(0.2)(attn2)

# --- Third Attention block for more capacity ---
attn3 = MultiHeadAttention(
	num_heads=8,
	key_dim=64,
	output_shape=384,
	dropout=0.1,
	name="attn_after_conv3"
)(attn2, attn2)
attn3 = AddLayer(name="resid_attn_conv3")([attn2, attn3])
attn3 = LayerNormalization()(attn3)
attn3 = Dropout(0.2)(attn3)

# Pooling: combine only average and attention-pooled outputs (no max pooling)
avg_pool = GlobalAveragePooling1D()(attn3)
attn_pool = Lambda(lambda t: tf.reduce_mean(t, axis=1))(attn3)
x = Concatenate()([avg_pool, attn_pool])



# --- Deep classification head: 2 dense layers, GELU activation, LayerNorm ---

# --- Dense(1024) + BatchNorm + Dropout ---
x = Dense(1024, activation='gelu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = LayerNormalization()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)


# Added Dense(128) layer

# --- Dense(512) + BatchNorm + Dropout ---
x = Dense(512, activation='gelu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = LayerNormalization()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

# --- Dense(256) + BatchNorm + Dropout ---
x = Dense(256, activation='gelu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = LayerNormalization()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)


# Final normalization before output
x = LayerNormalization()(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)


# Use AdamW optimizer if available, else fallback to Adam
try:
	from tensorflow.keras.optimizers import AdamW
	optimizer = AdamW(learning_rate=LEARNING_RATE)
except ImportError:
	optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(
	optimizer=optimizer,
	loss=focal_loss,
	metrics=['accuracy']
)

model.summary()

print(f"  Epochs: {EPOCHS}")
print(f"  Model will train for {EPOCHS} epochs (robust quick test mode)")

# ===================== TRAINING & CALLBACKS =====================
checkpoint_dir = '/root/models'
plot_dir = '/root/plots'
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

checkpoint_path = os.path.join(checkpoint_dir, 'attention_best.keras')
early_stop = EarlyStopping(monitor='val_accuracy', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
model_ckpt = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)


# --- Learning Rate Scheduler ---
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=7, min_lr=1e-6, verbose=1)

history = model.fit(
	X_train_aug, y_train_aug,
	batch_size=BACTH_SIZE,
	epochs=EPOCHS,
	validation_data=(X_val, y_val_onehot),
	callbacks=[early_stop, model_ckpt, reduce_lr],
	class_weight=class_weight_dict,
	verbose=1
)


# ===================== EVALUATION & CONFUSION MATRIX (TEST SET) =====================
from sklearn.metrics import classification_report
print("\nEvaluating on test set...")
X_test_proc = preprocess_sequences(test_data_raw, MAX_SEQ_LENGTH, NUM_FEATURES, uniform_sample=True)
y_test = np.array(test_labels_raw, dtype=np.int32)
y_pred_probs = model.predict(X_test_proc, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Per-class metrics
class_names = [LABEL_NAMES.get(i, str(i)) for i in range(NUM_CLASSES)]
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
import json
with open(os.path.join(plot_dir, 'classification_report.json'), 'w') as f:
	json.dump(report, f, indent=2)
with open(os.path.join(plot_dir, 'classification_report.txt'), 'w') as f:
	f.write(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
print(f"Classification report saved to {os.path.join(plot_dir, 'classification_report.txt')}")

# Normalized confusion matrix (percentages)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)), normalize='true')
fig, ax = plt.subplots(figsize=(18, 16))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation=90)
plt.title('Normalized Confusion Matrix (Test)')
plt.tight_layout()
cm_path = os.path.join(plot_dir, 'confusion_matrix_normalized.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Normalized confusion matrix saved to: {cm_path}")

# Also save raw confusion matrix
cm_raw = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
np.save(os.path.join(plot_dir, 'confusion_matrix_raw.npy'), cm_raw)
np.save(os.path.join(plot_dir, 'confusion_matrix_normalized.npy'), cm)
print("All per-class metrics and confusion matrices saved.")
