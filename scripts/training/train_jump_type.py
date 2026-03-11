"""
Stage 1: Jump type classifier.
Maps all jump classes to 6 types and all non-jump classes to Other.
"""

import os
import json
import pickle
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, BatchNormalization,
    Conv1D, Input, Multiply, Softmax, Lambda
)
from tensorflow.keras.layers import GlobalAveragePooling1D, Concatenate, Add as AddLayer
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import to_categorical

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = os.path.expanduser('~/skeleton/')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, 'models')
PLOT_SAVE_DIR = os.path.join(SCRIPT_DIR, 'plots')

MAX_SEQ_LENGTH = 500
NUM_KEYPOINTS = 17
NUM_COORDS = 3
NUM_FEATURES = NUM_KEYPOINTS * NUM_COORDS

DROPOUT_RATE = 0.3
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 0.0  # Use standard cross-entropy behavior

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
VALIDATION_SPLIT = 0.2

AUGMENTATION_ENABLED = False

JUMP_TYPES = ["Axel", "Flip", "Loop", "Lutz", "Salchow", "Toeloop"]

# ============================================================
# 2. DATA LOADING + LABEL MAPPING
# ============================================================

print("=" * 60)
print("STAGE 1: Jump Type Classification")
print("=" * 60)

with open(os.path.join(DATA_DIR, 'train_data.pkl'), 'rb') as f:
    train_data_raw = pickle.load(f)

with open(os.path.join(DATA_DIR, 'train_label.pkl'), 'rb') as f:
    train_labels_raw = pickle.load(f)

with open(os.path.join(DATA_DIR, 'test_data.pkl'), 'rb') as f:
    test_data_raw = pickle.load(f)

with open(os.path.join(DATA_DIR, 'test_label.pkl'), 'rb') as f:
    test_labels_raw = pickle.load(f)

with open(os.path.join(DATA_DIR, 'label_mapping.json'), 'r') as f:
    label_names = {int(k): v for k, v in json.load(f).items()}

print(f"  Loaded {len(train_data_raw)} training samples")
print(f"  Loaded {len(test_data_raw)} test samples")


def parse_jump_type(name):
    if not name or not name[0].isdigit():
        return None
    for jump in JUMP_TYPES:
        if jump in name:
            return jump
    return None


# Map labels: jump types -> 1..6, non-jumps -> 0 (Other)
label_to_jump = {label: parse_jump_type(name) for label, name in label_names.items()}

jump_type_to_idx = {jump: i + 1 for i, jump in enumerate(JUMP_TYPES)}
idx_to_jump_type = {0: "Other"}
idx_to_jump_type.update({i + 1: jump for i, jump in enumerate(JUMP_TYPES)})

train_labels_mapped = []
for label in train_labels_raw:
    jump_type = label_to_jump.get(label)
    train_labels_mapped.append(jump_type_to_idx.get(jump_type, 0))

test_labels_mapped = []
for label in test_labels_raw:
    jump_type = label_to_jump.get(label)
    test_labels_mapped.append(jump_type_to_idx.get(jump_type, 0))

print("\n  Label distribution (train):")
print(Counter(train_labels_mapped))
print("  Label distribution (test):")
print(Counter(test_labels_mapped))

NUM_CLASSES = len(idx_to_jump_type)

# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Preprocessing data...")
print("=" * 60)


def preprocess_sequences(data_list, max_len, num_features, uniform_sample=True):
    """Convert variable-length sequences into fixed-size arrays."""
    num_samples = len(data_list)
    processed = np.zeros((num_samples, max_len, num_features), dtype=np.float32)

    skipped = 0
    for i, sample in enumerate(data_list):
        sample = sample.squeeze(axis=-1)
        seq_len = sample.shape[0]
        if seq_len == 0:
            skipped += 1
            continue
        sample_flat = sample.reshape(seq_len, -1)

        if seq_len > max_len:
            if uniform_sample:
                indices = np.linspace(0, seq_len - 1, max_len, dtype=int)
                processed[i] = sample_flat[indices]
            else:
                processed[i] = sample_flat[:max_len]
        else:
            processed[i, :seq_len, :] = sample_flat

    if skipped > 0:
        print(f"  WARNING: {skipped} empty sequences encountered (filled with zeros)")
    return processed


X_train = preprocess_sequences(train_data_raw, MAX_SEQ_LENGTH, NUM_FEATURES, uniform_sample=True)
X_test = preprocess_sequences(test_data_raw, MAX_SEQ_LENGTH, NUM_FEATURES, uniform_sample=True)

# Label encoding
y_train = np.array(train_labels_mapped, dtype=np.int32)
y_test = np.array(test_labels_mapped, dtype=np.int32)

y_train_onehot = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_onehot = to_categorical(y_test, num_classes=NUM_CLASSES)

print(f"    y_train_onehot shape: {y_train_onehot.shape}")
print(f"    y_test_onehot shape: {y_test_onehot.shape}")

# Class weights
print("\n  Computing class weights for imbalanced data...")
class_weights_array = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights_array)}
print(f"    Class weight range: [{min(class_weight_dict.values()):.2f}, {max(class_weight_dict.values()):.2f}]")

# Normalization
print("\n  Normalizing data (vectorized)...")
nonzero_mask = np.any(X_train != 0, axis=2)
all_frames = X_train[nonzero_mask]
feature_mean = all_frames.mean(axis=0)
feature_std = all_frames.std(axis=0)
feature_std[feature_std < 1e-8] = 1.0

X_train_norm = (X_train - feature_mean) / feature_std
train_pad_mask = ~np.any(X_train != 0, axis=2, keepdims=True)
X_train_norm[np.broadcast_to(train_pad_mask, X_train_norm.shape)] = 0.0
X_train = X_train_norm

X_test_norm = (X_test - feature_mean) / feature_std
test_pad_mask = ~np.any(X_test != 0, axis=2, keepdims=True)
X_test_norm[np.broadcast_to(test_pad_mask, X_test_norm.shape)] = 0.0
X_test = X_test_norm

# Velocity features
train_real_mask = np.any(X_train != 0, axis=2, keepdims=True)
X_train_vel = np.diff(X_train, axis=1, prepend=X_train[:, :1, :])
X_train_vel *= train_real_mask
X_train = np.concatenate([X_train, X_train_vel], axis=-1)

test_real_mask = np.any(X_test != 0, axis=2, keepdims=True)
X_test_vel = np.diff(X_test, axis=1, prepend=X_test[:, :1, :])
X_test_vel *= test_real_mask
X_test = np.concatenate([X_test, X_test_vel], axis=-1)

# ============================================================
# 4. MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Building the model...")
print("=" * 60)


def focal_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    y_smooth = y_true * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / NUM_CLASSES
    ce = -y_smooth * tf.math.log(y_pred)
    weight = tf.pow(1 - y_pred, FOCAL_GAMMA)
    return tf.reduce_sum(weight * ce, axis=-1)


input_features = X_train.shape[-1]
inputs = Input(shape=(MAX_SEQ_LENGTH, input_features))


def conv_block(x, filters, kernel_size, dropout_rate=0.15):
    """Convolutional block with batch norm and residual connection."""
    shortcut = Conv1D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
    x = Conv1D(filters, kernel_size, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = AddLayer()([shortcut, x])
    return x


x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
x = BatchNormalization()(x)

x = conv_block(x, 128, 3)
x = conv_block(x, 128, 3)
x = conv_block(x, 256, 5)

lstm1 = Bidirectional(LSTM(128, return_sequences=True))(x)
lstm1 = BatchNormalization()(lstm1)
lstm1 = Dropout(DROPOUT_RATE)(lstm1)

lstm2 = Bidirectional(LSTM(128, return_sequences=True))(lstm1)
lstm2 = BatchNormalization()(lstm2)
lstm2 = Dropout(DROPOUT_RATE)(lstm2)

lstm_out = AddLayer()([lstm1, lstm2])

attn_global = Dense(1)(lstm_out)
attn_weights_global = Softmax(axis=1)(attn_global)
attn_output_global = Multiply()([lstm_out, attn_weights_global])

attn_local = Dense(1, activation='tanh')(lstm_out)
attn_weights_local = Softmax(axis=1)(attn_local)
attn_output_local = Multiply()([lstm_out, attn_weights_local])

max_pool = Lambda(lambda t: tf.reduce_max(t, axis=1))(lstm_out)
avg_pool = GlobalAveragePooling1D()(lstm_out)
attn_pool_global = Lambda(lambda t: tf.reduce_sum(t, axis=1))(attn_output_global)
attn_pool_local = Lambda(lambda t: tf.reduce_sum(t, axis=1))(attn_output_local)

x = Concatenate()([max_pool, avg_pool, attn_pool_global, attn_pool_local])

x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=focal_loss, metrics=['accuracy'])
model.summary()

# ============================================================
# 5. CALLBACKS
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Setting up training callbacks...")
print("=" * 60)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def lr_schedule(epoch, lr):
    return LEARNING_RATE


callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_DIR, 'jump_type_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    LearningRateScheduler(lr_schedule, verbose=1)
]

# ============================================================
# 6. TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Training the model...")
print("=" * 60)

history = model.fit(
    X_train, y_train_onehot,
    validation_split=VALIDATION_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================================
# 7. EVALUATE + SAVE
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Evaluating the model...")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
print(f"\n  Final Test Loss:     {test_loss:.4f}")
print(f"  Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

final_model_path = os.path.join(MODEL_SAVE_DIR, 'jump_type_final.keras')
model.save(final_model_path)
print(f"  Final model saved to: {final_model_path}")

history_path = os.path.join(MODEL_SAVE_DIR, 'jump_type_history.pkl')
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
print(f"  Training history saved to: {history_path}")

# Classification report
class_names = [idx_to_jump_type[i] for i in range(NUM_CLASSES)]
report = classification_report(y_test, np.argmax(model.predict(X_test, verbose=0), axis=1),
                               target_names=class_names, zero_division=0)
print("\n  Classification Report:")
print(report)

report_path = os.path.join(PLOT_SAVE_DIR, 'jump_type_classification_report.txt')
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
with open(report_path, 'w') as f:
    f.write(report)
print(f"  Classification report saved to: {report_path}")

# Confusion matrix
cm = confusion_matrix(y_test, np.argmax(model.predict(X_test, verbose=0), axis=1),
                      labels=list(range(NUM_CLASSES)))

cm_path = os.path.join(PLOT_SAVE_DIR, 'jump_type_confusion_matrix.npy')
np.save(cm_path, cm)
print(f"  Confusion matrix saved to: {cm_path}")
