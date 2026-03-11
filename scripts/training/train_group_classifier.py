"""
Script to filter the dataset for a specific group (e.g., jumps, spins, steps), relabel to contiguous indices, and train a simple classifier.
Edit the GROUP variable to select which group to train on.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Config ---
DATA_DIR = Path.home() / "AI_Skating_Coach_Dataset_Clean"
GROUP = "jumps"  # Options: "jumps", "spins", "steps", "choreo", etc.

# --- Define class groupings ---
with open(DATA_DIR / "label_mapping.json", 'r') as f:
    label_mapping = json.load(f)
    LABEL_NAMES = {int(k): v for k, v in label_mapping.items()}

group_defs = {
    "jumps":   [k for k, v in LABEL_NAMES.items() if any(j in v for j in ["Axel", "Flip", "Loop", "Lutz", "Salchow", "Toeloop"])],
    "spins":   [k for k, v in LABEL_NAMES.items() if "Sp" in v],
    "steps":   [k for k, v in LABEL_NAMES.items() if "StepSeq" in v],
    "choreo":  [k for k, v in LABEL_NAMES.items() if "Chore" in v or "ChCombo" in v],
}

if GROUP not in group_defs:
    raise ValueError(f"Unknown group: {GROUP}")

selected_labels = set(group_defs[GROUP])

# --- Load and filter data ---
with open(DATA_DIR / "train_data.pkl", 'rb') as f:
    X_train = pickle.load(f)
with open(DATA_DIR / "train_label.pkl", 'rb') as f:
    y_train = pickle.load(f)
with open(DATA_DIR / "test_data.pkl", 'rb') as f:
    X_test = pickle.load(f)
with open(DATA_DIR / "test_label.pkl", 'rb') as f:
    y_test = pickle.load(f)

# Filter for selected group
X_train_filt, y_train_filt = zip(*[(x, y) for x, y in zip(X_train, y_train) if y in selected_labels])
X_test_filt, y_test_filt = zip(*[(x, y) for x, y in zip(X_test, y_test) if y in selected_labels])

# Relabel to contiguous indices
unique_labels = sorted(set(y_train_filt) | set(y_test_filt))
label_to_new = {old: new for new, old in enumerate(unique_labels)}
new_to_label = {v: k for k, v in label_to_new.items()}
y_train_new = np.array([label_to_new[y] for y in y_train_filt])
y_test_new = np.array([label_to_new[y] for y in y_test_filt])

# --- Prepare data for training ---
X_train_arr = np.array(X_train_filt)
X_test_arr = np.array(X_test_filt)
num_classes = len(unique_labels)
y_train_cat = to_categorical(y_train_new, num_classes)
y_test_cat = to_categorical(y_test_new, num_classes)

# --- Simple model (edit as needed) ---
model = Sequential([
    Flatten(input_shape=X_train_arr.shape[1:]),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Train ---
callbacks = [EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)]
history = model.fit(
    X_train_arr, y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=2
)

# --- Evaluate ---
loss, acc = model.evaluate(X_test_arr, y_test_cat, verbose=0)
print(f"Test accuracy for {GROUP}: {acc*100:.2f}%")

# --- Save model and label mapping ---
model.save(f"model_{GROUP}.keras")
with open(f"label_map_{GROUP}.json", "w") as f:
    json.dump({str(i): LABEL_NAMES[new_to_label[i]] for i in range(num_classes)}, f, indent=2)
print(f"Model and label map saved for {GROUP}.")
