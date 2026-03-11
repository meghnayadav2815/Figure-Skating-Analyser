# Data Pipeline: Current Dataset + Skating Verse

**Goal:** Merge your existing skating action dataset with Skating Verse videos to improve accuracy by addressing class imbalance (especially rare jumps: Flip, Salchow).

---

## **Pipeline Architecture**

```
Your Data (pickle)              Skating Verse Videos (raw)
       ↓                                 ↓
   [Load]                        [Extract Skeletons]
       ↓                                 ↓
       └──────────  [Standardize]  ─────┘
                         ↓
                  [Map Labels]
                         ↓
                  [Merge Datasets]
                         ↓
                [Balance/Augment]
                         ↓
                   [Export Pickle]
                         ↓
              merged_train_data.pkl (ready to train)
```

---

## **Quick Start**

### **1. Install Dependencies**

```bash
pip install -r data_pipeline/requirements.txt
```

### **2. Prepare Directories**

```bash
mkdir -p datasets/{current,skating_verse,merged}
```

### **3. Add Data**

```bash
# Copy your current data to datasets/current/
cp train_data.pkl datasets/current/
cp train_label.pkl datasets/current/
cp test_data.pkl datasets/current/
cp test_label.pkl datasets/current/
cp label_mapping.json datasets/current/

# Download Skating Verse videos to datasets/skating_verse/
# (Videos will be auto-processed by the pipeline)
```

### **4. Run Pipeline**

```bash
cd data_pipeline
python pipeline.py
```

Output files appear in `datasets/merged/`:
- `merged_train_data.pkl` (ready to train)
- `merged_train_labels.pkl`
- `merged_test_data.pkl`
- `merged_test_labels.pkl`
- `label_mapping.json`
- `metadata.json`

---

## **Detailed Stage Breakdown**

### **Stage 1: Data Ingestion**
**What it does:**
- Loads your current pickle dataset (train/test splits)
- Loads your label mapping (19 classes)
- Lists all Skating Verse video files

**Config:** `config.py` → Load paths
**Output:** Loaded data in memory

**Code:**
```python
from ingestion import DataIngestor

ingestor = DataIngestor()
ingestor.load_current_dataset()
ingestor.load_label_mapping()
sv_videos = ingestor.list_skating_verse_videos()
```

---

### **Stage 2: Skeleton Extraction (Skating Verse Only)**
**What it does:**
- Extracts 17-joint COCO skeletons from raw Skating Verse videos
- Handles video errors, missing frames
- Pads/truncates sequences to 500 frames
- Saves extracted skeletons to cache

**Config:** `config.py` → `SKELETON_PARAMS` 
- Extractor: MediaPipe (recommended) or OpenPose
- Input: Raw video files
- Output: (500, 17*3) = (500, 51) arrays (compatible with current dataset)

**Code:**
```python
from skeleton_extraction import SkeletonExtractor

extractor = SkeletonExtractor(extractor_type="mediapipe")
skeletons, valid_indices = extractor.batch_extract(video_paths)
# skeletons shape: (n_videos, 500, 51)
```

**Performance:**
- ~2-3 min per video (GPU accelerated with MediaPipe)
- 100 videos ≈ 3-5 hours
- Cache saves time on reruns

---

### **Stage 3: Data Standardization**
**What it does:**
- Normalizes both datasets to same scale
- Handles missing/low-confidence detections
- Smooths skeleton trajectories (reduces jitter)
- Validates data quality

**Config:** `config.py` → `NORM_PARAMS`
- Normalize: T/F
- Mean-center joints: T/F
- Std-normalize: T/F

**Code:**
```python
from standardization import DataStandardizer

standardizer = DataStandardizer()
standardizer.fit_normalization(all_skeletons)  # Fit on combined data
standardized = standardizer.batch_standardize(all_skeletons)
```

**Key operations:**
```
1. Per-joint affine normalization (x,y coordinates)
2. Interpolate low-confidence frames
3. Smooth with moving average (window=3)
4. Validate: check NaN, shape, confidence levels
```

---

### **Stage 4: Label Mapping**
**What it does:**
- Maps Skating Verse labels → your 19 classes
- Handles label variants (e.g., 1Axel, 2Axel, 3Axel → Axel)
- Reports unmapped labels

**Config:** `config.py` → `LABEL_MAPPING`

**Example mapping:**
```python
LABEL_MAPPING = {
    # Your current 19 classes
    "Axel": "Axel",
    "Flip": "Flip",
    # ... etc
    
    # Skating Verse variants → Your format
    "1Axel": "Axel",
    "2Axel": "Axel",
    "3Axel": "Axel",
    "1Flip": "Flip",
    # ... etc
}
```

**Code:**
```python
from label_mapping import LabelMapper

mapper = LabelMapper(LABEL_MAPPING)
mapped_labels = mapper.batch_map_labels(sv_labels)
```

---

### **Stage 5: Dataset Merging**
**What it does:**
- Concatenates standardized data from both sources
- Merges label arrays
- Preserves test set (unchanged)

**Code:**
```python
merged_train_data = np.vstack([current_train_std, sv_std])
merged_train_labels = np.concatenate([current_labels, sv_labels])
```

**Result:**
- Input: Current (3.7k) + Skating Verse (~X videos)
- Output: Merged (3.7k + X) samples

---

### **Stage 6: Balancing & Augmentation**
**What it does:**
- Identifies rare classes (default: <50 samples)
- Applies oversampling + augmentation
  - Mirror (flip skeleton left-right)
  - Noise (add Gaussian noise)
  - Scale (temporal/spatial scaling)

**Config:** `config.py` → `AUGMENTATION_PARAMS`
```python
{
    "oversample_threshold": 50,  # Classes with <50 get boosted
    "augmentation_ratio": 4,     # 4x augmentation per sample
    "augmentation_types": ["mirror", "noise", "scale"],
}
```

**Result:**
- Before: Flip=82 samples, Salchow=58 samples → imbalanced
- After: All classes ~300+ samples → balanced

---

### **Stage 7: Export**
**What it does:**
- Saves merged dataset as pickle files
- Saves label mapping (JSON)
- Saves metadata

**Outputs:**
```
datasets/merged/
├── merged_train_data.pkl       # (n_train, 500, 51) - 17 joints × 3 features
├── merged_train_labels.pkl     # (n_train,)
├── merged_test_data.pkl        # (n_test, 500, 51) - 17 joints × 3 features
├── merged_test_labels.pkl      # (n_test,)
├── label_mapping.json          # {0: "Axel", 1: "Flip", ...}
└── metadata.json               # {dataset_size, skeleton_shape, ...}
```

---

## **How to Use Merged Data for Training**

Once the pipeline completes, load merged data **instead of** your original data:

```python
import pickle
import numpy as np
from pathlib import Path

merged_dir = Path("datasets/merged")

# Load merged data
with open(merged_dir / "merged_train_data.pkl", 'rb') as f:
    train_data = pickle.load(f)
with open(merged_dir / "merged_train_labels.pkl", 'rb') as f:
    train_label = pickle.load(f)

# Use with your training script
# (No other code changes needed!)
model.fit(train_data, train_label, ...)
```

---

## **Configuration Guide**

Edit `config.py` to customize the pipeline:

### **Skeleton Extraction**
```python
SKELETON_PARAMS = {
    "num_joints": 17,              # COCO format (downsampled from MediaPipe's 33)
    "sequence_length": 500,        # Frame length
    "confidence_threshold": 0.5,   # Min confidence for keeping joint
    "extractor": "mediapipe",      # or "openpose"
}
```

### **Label Mapping**
```python
LABEL_MAPPING = {
    "Axel": "Axel",          # Format: "SV_label": "Your_label"
    "1Axel": "Axel",         # Variants map to same class
    # ... add all Skating Verse labels here
}
```

### **Augmentation**
```python
AUGMENTATION_PARAMS = {
    "enable": True,
    "oversample_threshold": 50,       # Classes with <50 samples get boosted
    "augmentation_ratio": 4,          # 4x augmentation
    "augmentation_types": ["mirror", "noise", "scale"],
    "noise_std": 0.02,
    "scale_range": (0.90, 1.10),      # 10% temporal scaling
}
```

---

## **Troubleshooting**

### **Issue: "No Skating Verse videos found"**
- Check that videos are in `datasets/skating_verse/`
- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

### **Issue: "Shape mismatch" for Skating Verse**
- Ensure your extractor produces 51-joint format
- MediaPipe outputs 33 joints → may need mapping to 51

### **Issue: Memory error during extraction**
- Process videos in smaller batches
- Edit `pipeline.py` line in `step_2_extract_skating_verse_skeletons()`:
  ```python
  batch_size = 10  # Process 10 at a time
  for i in range(0, len(videos), batch_size):
      batch = videos[i:i+batch_size]
      skeletons, _ = extractor.batch_extract(batch)
  ```

### **Issue: Unmapped labels**
- Add missing labels to `LABEL_MAPPING` in `config.py`
- Rerun pipeline

---

## **Performance Metrics**

Expected runtime on RTX A6000:

| Stage | Time (100 videos) | Bottleneck |
|-------|-----------------|-----------|
| Ingestion | ~10 sec | I/O |
| Extraction | 3-5 hours | MediaPipe inference |
| Standardization | ~2 min | NumPy operations |
| Label Mapping | <1 sec | Dictionary lookup |
| Merging | <1 sec | NumPy concat |
| Balancing | ~5 min | Augmentation |
| Export | ~30 sec | Pickle I/O |
| **Total** | **~3.5-5.5 hours** | Extraction |

**Optimization:** Run extraction overnight, use cached skeletons for iterations.

---

## **Next Steps**

1. **Download Skating Verse data:** [Kaggle link](https://www.kaggle.com/datasets/mdnau6/skitingverse-annotation)
2. **Configure label mapping:** Edit `LABEL_MAPPING` in `config.py`
3. **Run pipeline:** `python pipeline.py`
4. **Train model:** Use `merged_train_data.pkl` with your training script

---

## **API Reference**

### `DataPipeline.run()`
Runs the complete pipeline end-to-end.

```python
from pipeline import DataPipeline

pipeline = DataPipeline()
pipeline.run()  # Returns: bool (success/failure)
```

### Individual stages (advanced use)
```python
pipeline = DataPipeline()
pipeline.step_1_ingest_data()
pipeline.step_2_extract_skating_verse_skeletons()
# ... etc
```

---

## **File Structure**

```
data_pipeline/
├── config.py                    # Configuration (edit this!)
├── ingestion.py                 # Load datasets
├── skeleton_extraction.py        # Extract skeletons from videos
├── standardization.py            # Normalize data
├── label_mapping.py              # Map labels
├── pipeline.py                   # Main orchestrator
├── __init__.py                   # Package init
└── README.md                     # This file
```

---

**Questions?** See the code comments or check individual module docstrings.
