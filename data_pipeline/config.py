"""
Data Pipeline Configuration
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Head dataset (main storage location)
HEAD_DIR = DATA_DIR / "head"

# Source data (input)
CURRENT_DATA_PATH = DATA_DIR / "current"  # Where user puts their current data
SKATING_VERSE_PATH = DATA_DIR / "sv"      # Where user puts SV videos

# Output organization under head/
HEAD_CURRENT_DIR = HEAD_DIR / "current"
HEAD_SV_DIR = HEAD_DIR / "sv"
HEAD_MERGED_DIR = HEAD_DIR / "merged"

# Action categories for final storage
ACTION_CATEGORIES = {
    "jumps": ["Axel", "Flip", "Loop", "Lutz", "Salchow"],
    "spins": ["ChCamelSp3", "ChCamelSp4", "ChComboSp2", "ChComboSp3", "ChComboSp4", "ChSitSp4", "FlySitSp3", "LaybackSp3", "LaybackSp4"],
    "steps": ["StepSeq1", "StepSeq2", "StepSeq3", "ChoreSeq"],
}

# Create directories if they don't exist
for action in ACTION_CATEGORIES.keys():
    (HEAD_CURRENT_DIR / action).mkdir(parents=True, exist_ok=True)
    (HEAD_SV_DIR / action).mkdir(parents=True, exist_ok=True)
    (HEAD_MERGED_DIR / action).mkdir(parents=True, exist_ok=True)

# Skeleton extraction
SKELETON_PARAMS = {
    "num_joints": 17,  # COCO format: 17 keypoints (compatible with current data)
    "sequence_length": 500,
    "confidence_threshold": 0.5,  # MediaPipe confidence score
    "extractor": "mediapipe",     # Extracts from MediaPipe's 33 joints, downsampled to COCO 17
    # COCO 17 keypoints: nose, eyes(2), ears(2), shoulders(2), elbows(2), wrists(2), hips(2), knees(2), ankles(2)
}

# Data normalization
NORM_PARAMS = {
    "normalize": True,
    "mean_center": True,
    "std_normalize": True,
}

# Label mapping: Skating Verse → Your 19 classes
LABEL_MAPPING = {
    # Your current 19 classes
    "Axel": "Axel",
    "Flip": "Flip",
    "Loop": "Loop",
    "Lutz": "Lutz",
    "Salchow": "Salchow",
    "ChCamelSp3": "ChCamelSp3",
    "ChCamelSp4": "ChCamelSp4",
    "ChComboSp2": "ChComboSp2",
    "ChComboSp3": "ChComboSp3",
    "ChComboSp4": "ChComboSp4",
    "ChoreSeq1": "ChoreSeq1",
    "ChSitSp4": "ChSitSp4",
    "FlySitSp3": "FlySitSp3",
    "LaybackSp3": "LaybackSp3",
    "LaybackSp4": "LaybackSp4",
    "StepSeq1": "StepSeq1",
    "StepSeq2": "StepSeq2",
    "StepSeq3": "StepSeq3",
    "StepSeq4": "StepSeq4",
    
    # Skating Verse mappings (adjust based on actual Skating Verse labels)
    # Example: if Skating Verse has "1Axel", "2Axel", "3Axel" → all map to "Axel"
    "1Axel": "Axel",
    "2Axel": "Axel",
    "3Axel": "Axel",
    "1Flip": "Flip",
    "2Flip": "Flip",
    "3Flip": "Flip",
    "1Loop": "Loop",
    "2Loop": "Loop",
    "3Loop": "Loop",
    "1Lutz": "Lutz",
    "2Lutz": "Lutz",
    "3Lutz": "Lutz",
    "1Salchow": "Salchow",
    "2Salchow": "Salchow",
    "3Salchow": "Salchow",
    # Add Skating Verse spin/step labels as needed
}

# Data split
SPLIT_PARAMS = {
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,
    "stratify": True,  # Maintain class distribution
}

# Augmentation for rare classes
AUGMENTATION_PARAMS = {
    "enable": True,
    "oversample_threshold": 50,  # Classes with <50 samples get oversampled
    "augmentation_ratio": 4,      # 4x augmentation
    "augmentation_types": ["mirror", "noise", "scale"],
    "noise_std": 0.02,
    "scale_range": (0.90, 1.10),
}

# Output
OUTPUT_PARAMS = {
    "format": "pickle",
    "compression": True,
    "save_metadata": True,
}
