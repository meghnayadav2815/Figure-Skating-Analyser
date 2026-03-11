"""
BINARY JUMP DETECTION
Simplified approach: Distinguish "Jump" vs "Non-Jump" (spins, spirals, sequences)
This is learnable from skeleton data (velocity/timing patterns).
Jump TYPE is not learnable, so we skip that stage entirely.
"""

import os
import json
import pickle
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = os.path.expanduser("~/skeleton/")
MODEL_DIR = "/root/AI_Skating_Coach/models"
PLOT_DIR = "/root/AI_Skating_Coach/plots"
LABEL_MAPPING_FILE = os.path.join(DATA_DIR, "label_mapping.json")

MAX_SEQ_LENGTH = 500
NUM_KEYPOINTS = 17
NUM_COORDS = 3
NUM_FEATURES = NUM_KEYPOINTS * NUM_COORDS

DROPOUT_RATE = 0.3
LABEL_SMOOTHING = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10
VALIDATION_SPLIT = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 60)
print("BINARY JUMP DETECTION")
print("=" * 60)

# ============================================================
# STEP 1: Load data
# ============================================================
print("\n  STEP 1: Loading data...")

# Load raw data from pickle files
with open(os.path.join(DATA_DIR, "train_data.pkl"), "rb") as f:
    train_data_raw = pickle.load(f)
with open(os.path.join(DATA_DIR, "train_label.pkl"), "rb") as f:
    train_labels_raw = pickle.load(f)
with open(os.path.join(DATA_DIR, "test_data.pkl"), "rb") as f:
    test_data_raw = pickle.load(f)
with open(os.path.join(DATA_DIR, "test_label.pkl"), "rb") as f:
    test_labels_raw = pickle.load(f)

with open(os.path.join(DATA_DIR, "label_mapping.json"), "r") as f:
    label_names = {int(k): v for k, v in json.load(f).items()}

print(f"  Loaded {len(train_data_raw)} training samples")
print(f"  Loaded {len(test_data_raw)} test samples")

def is_jump_class(class_name):
    """Check if class name is a jump (starts with digit)"""
    return class_name and class_name[0].isdigit()

# Map raw labels to binary labels
train_labels_binary = []
for label in train_labels_raw:
    class_name = label_names.get(label, "")
    binary_label = 1 if is_jump_class(class_name) else 0
    train_labels_binary.append(binary_label)

test_labels_binary = []
for label in test_labels_raw:
    class_name = label_names.get(label, "")
    binary_label = 1 if is_jump_class(class_name) else 0
    test_labels_binary.append(binary_label)

X_train = train_data_raw
y_train = np.array(train_labels_binary)
X_test = test_data_raw
y_test = np.array(test_labels_binary)

print(f"  Loaded {len(X_train)} training samples")
print(f"  Loaded {len(X_test)} test samples")
print(f"\n  Label distribution (train):")
print(f"  {Counter(y_train)}")
print(f"  Label distribution (test):")
print(f"  {Counter(y_test)}")

# ============================================================
# STEP 2: Preprocessing
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

# Preprocess sequences
X_train = preprocess_sequences(X_train, MAX_SEQ_LENGTH, NUM_FEATURES)
X_test = preprocess_sequences(X_test, MAX_SEQ_LENGTH, NUM_FEATURES)

# Compute velocity features
def add_velocity_features(X):
    """Append velocity features to position features"""
    X_vel = np.diff(X, axis=1, prepend=X[:, 0:1, :])
    return np.concatenate([X, X_vel], axis=-1)

X_train = add_velocity_features(X_train)
X_test = add_velocity_features(X_test)

# One-hot encode labels (binary: 2 classes)
y_train_onehot = keras.utils.to_categorical(y_train, num_classes=2)
y_test_onehot = keras.utils.to_categorical(y_test, num_classes=2)

print(f"  X_train shape after preprocessing: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_train_onehot shape: {y_train_onehot.shape}")
print(f"  y_test_onehot shape: {y_test_onehot.shape}")

# Compute class weights
class_weights = compute_class_weight(
    "balanced", 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"  Class weights: {class_weight_dict}")

# Standardize data
scaler = StandardScaler()
X_train_flat = X_train.reshape(len(X_train), -1)
X_train_flat = scaler.fit_transform(X_train_flat)
X_train = X_train_flat.reshape(X_train.shape)

X_test_flat = X_test.reshape(len(X_test), -1)
X_test_flat = scaler.transform(X_test_flat)
X_test = X_test_flat.reshape(X_test.shape)

print(f"  Data normalized and standardized")

# ============================================================
# STEP 3: Build model (simplified for binary task)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Building the model...")
print("=" * 60)

def build_jump_detection_model(input_shape, num_classes=2):
    """Simplified architecture for binary classification"""
    inputs = layers.Input(shape=input_shape)
    
    # Conv blocks (simpler than original)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # BiLSTM
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Dense head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_jump_detection_model((MAX_SEQ_LENGTH, X_train.shape[2]))
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# STEP 4: Training
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Training the model...")
print("=" * 60)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, "jump_detection_best.keras"),
    monitor="val_accuracy",
    mode="max",
    save_best_only=True
)

history = model.fit(
    X_train, y_train_onehot,
    validation_split=VALIDATION_SPLIT,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping, model_checkpoint],
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================================
# STEP 5: Evaluation
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Evaluating the model...")
print("=" * 60)

test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
print(f"\n  Test Loss:     {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Save final model
model.save(os.path.join(MODEL_DIR, "jump_detection_final.keras"))
with open(os.path.join(MODEL_DIR, "jump_detection_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

# Predictions and classification report
y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
y_test_labels = y_test

print(f"\n  Classification Report:")
print(classification_report(y_test_labels, y_pred, 
                          target_names=["Non-Jump (Spin/Spiral/Seq)", "Jump"]))

# Save classification report
with open(os.path.join(PLOT_DIR, "jump_detection_classification_report.txt"), "w") as f:
    f.write(classification_report(y_test_labels, y_pred, 
                                 target_names=["Non-Jump", "Jump"]))

# Save confusion matrix
cm = confusion_matrix(y_test_labels, y_pred)
np.save(os.path.join(PLOT_DIR, "jump_detection_confusion_matrix.npy"), cm)

print(f"\n  Confusion Matrix:\n{cm}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Jump Detection - Training Accuracy")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Jump Detection - Training Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "jump_detection_training_history.png"), dpi=100)
print(f"\n  Training history plot saved.")

print("\n" + "=" * 60)
print("JUMP DETECTION COMPLETE")
print("=" * 60)
