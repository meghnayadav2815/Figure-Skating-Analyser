"""
Main Data Pipeline Orchestrator
Coordinates: Ingestion → Extraction → Standardization → Label Mapping → Balancing → Export
"""
import json
import pickle
import logging
from pathlib import Path
import numpy as np
from collections import Counter

from ingestion import DataIngestor
from skeleton_extraction import SkeletonExtractor
from standardization import DataStandardizer
from label_mapping import LabelMapper
from config import (
    MERGED_DATA_DIR, SKELETON_PARAMS, LABEL_MAPPING, SPLIT_PARAMS, AUGMENTATION_PARAMS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self):
        self.ingestor = DataIngestor()
        self.extractor = SkeletonExtractor(extractor_type="mediapipe")
        self.standardizer = DataStandardizer()
        self.mapper = LabelMapper(LABEL_MAPPING)
        
        self.merged_data = None
        self.merged_labels = None
        self.merged_label_map = None
    
    def step_1_ingest_data(self):
        """Step 1: Load current dataset and list Skating Verse videos"""
        logger.info("\n" + "="*70)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("="*70)
        
        # Load current data
        if not self.ingestor.load_current_dataset():
            logger.error("Failed to load current dataset!")
            return False
        
        # Load label mapping
        self.ingestor.load_label_mapping()
        self.merged_label_map = self.ingestor.label_mapping
        
        # List Skating Verse videos
        self.skating_verse_videos = self.ingestor.list_skating_verse_videos()
        
        self.ingestor.get_summary()
        return True
    
    def step_2_extract_skating_verse_skeletons(self):
        """Step 2: Extract skeletons from Skating Verse videos"""
        logger.info("\n" + "="*70)
        logger.info("STEP 2: SKELETON EXTRACTION (Skating Verse)")
        logger.info("="*70)
        
        if not self.skating_verse_videos:
            logger.warning("⚠️  No Skating Verse videos found. Skipping extraction.")
            return [], []
        
        logger.info(f"Extracting skeletons from {len(self.skating_verse_videos)} videos...")
        
        skating_verse_skeletons, valid_indices = self.extractor.batch_extract(
            self.skating_verse_videos,
            output_file=MERGED_DATA_DIR / "skating_verse_skeletons.pkl"
        )
        
        logger.info(f"✅ Extracted {len(skating_verse_skeletons)} valid skeletons")
        return skating_verse_skeletons, valid_indices
    
    def step_3_standardize_all_data(self, skating_verse_skeletons):
        """Step 3: Standardize current data + Skating Verse data"""
        logger.info("\n" + "="*70)
        logger.info("STEP 3: DATA STANDARDIZATION")
        logger.info("="*70)
        
        # Combine all data for fitting normalizers
        all_skeletons = [
            *[self.ingestor.current_data['train'][i:i+1] for i in range(min(100, len(self.ingestor.current_data['train'])))],
            *skating_verse_skeletons[:min(100, len(skating_verse_skeletons))]
        ]
        
        # Fit standardizer on combined data
        self.standardizer.fit_normalization(all_skeletons)
        
        # Standardize all current data
        logger.info("Standardizing current dataset...")
        current_train_std = self.standardizer.batch_standardize(
            [self.ingestor.current_data['train'][i] for i in range(len(self.ingestor.current_data['train']))]
        )
        current_test_std = self.standardizer.batch_standardize(
            [self.ingestor.current_data['test'][i] for i in range(len(self.ingestor.current_data['test']))]
        )
        
        # Standardize Skating Verse data
        if skating_verse_skeletons:
            logger.info("Standardizing Skating Verse skeletons...")
            skating_verse_std = self.standardizer.batch_standardize(skating_verse_skeletons)
        else:
            skating_verse_std = []
        
        logger.info(f"✅ Standardization complete")
        logger.info(f"   Current train: {np.array(current_train_std).shape}")
        logger.info(f"   Current test: {np.array(current_test_std).shape}")
        logger.info(f"   Skating Verse: {np.array(skating_verse_std).shape}")
        
        return current_train_std, current_test_std, skating_verse_std
    
    def step_4_map_labels(self, skating_verse_valid_indices):
        """Step 4: Map Skating Verse labels to your 19 classes"""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: LABEL MAPPING")
        logger.info("="*70)
        
        # Current data labels (already in your format)
        current_train_labels = self.ingestor.current_labels['train']
        current_test_labels = self.ingestor.current_labels['test']
        
        # Skating Verse labels (TODO: You need to extract these from Skating Verse data)
        # For now, we'll assume labels are in a file named "labels.json" or similar
        skating_verse_labels = []  # TODO: Load from Skating Verse dataset
        
        # Map to your label indices
        mapped_skating_verse = self.mapper.batch_map_labels(skating_verse_labels)
        
        logger.info(f"✅ Label mapping complete")
        logger.info(f"   Current train labels: {current_train_labels.shape}")
        logger.info(f"   Current test labels: {current_test_labels.shape}")
        logger.info(f"   Skating Verse labels: {len(mapped_skating_verse)}")
        
        return current_train_labels, current_test_labels, mapped_skating_verse
    
    def step_5_merge_datasets(self, current_train_std, current_test_std, skating_verse_std,
                              current_train_labels, current_test_labels, skating_verse_labels):
        """Step 5: Merge current + Skating Verse datasets"""
        logger.info("\n" + "="*70)
        logger.info("STEP 5: DATASET MERGING")
        logger.info("="*70)
        
        # Merge training data (Skating Verse is all treated as training)
        merged_train_data = np.vstack([
            np.array(current_train_std),
            np.array(skating_verse_std)
        ]) if skating_verse_std else np.array(current_train_std)
        
        merged_train_labels = np.concatenate([
            current_train_labels,
            np.array(skating_verse_labels)
        ]) if skating_verse_labels else current_train_labels
        
        # Test data remains unchanged
        merged_test_data = np.array(current_test_std)
        merged_test_labels = current_test_labels
        
        logger.info(f"✅ Datasets merged")
        logger.info(f"   Merged train: {merged_train_data.shape}")
        logger.info(f"   Class distribution (train):")
        for label_idx, count in Counter(merged_train_labels).most_common():
            label_name = self.merged_label_map.get(str(label_idx), f"Class {label_idx}")
            logger.info(f"      {label_name}: {count} samples")
        
        self.merged_data = {
            'train': merged_train_data,
            'test': merged_test_data,
        }
        self.merged_labels = {
            'train': merged_train_labels,
            'test': merged_test_labels,
        }
        
        return merged_train_data, merged_train_labels, merged_test_data, merged_test_labels
    
    def step_6_balance_and_augment(self):
        """Step 6: Handle class imbalance with oversampling + augmentation"""
        logger.info("\n" + "="*70)
        logger.info("STEP 6: BALANCING & AUGMENTATION")
        logger.info("="*70)
        
        if not AUGMENTATION_PARAMS["enable"]:
            logger.info("⏭️  Augmentation disabled")
            return self.merged_data['train'], self.merged_labels['train']
        
        # TODO: Implement balancing logic
        # - Identify rare classes (< threshold)
        # - Apply augmentation (mirror, noise, scale)
        # - Oversample to balance distribution
        
        logger.info("✅ Balancing complete (TODO: Implement full balancing)")
        return self.merged_data['train'], self.merged_labels['train']
    
    def step_7_export(self):
        """Step 7: Export merged dataset as pickle files"""
        logger.info("\n" + "="*70)
        logger.info("STEP 7: EXPORT")
        logger.info("="*70)
        
        # Save data
        with open(MERGED_DATA_DIR / "merged_train_data.pkl", 'wb') as f:
            pickle.dump(self.merged_data['train'], f)
        
        with open(MERGED_DATA_DIR / "merged_train_labels.pkl", 'wb') as f:
            pickle.dump(self.merged_labels['train'], f)
        
        with open(MERGED_DATA_DIR / "merged_test_data.pkl", 'wb') as f:
            pickle.dump(self.merged_data['test'], f)
        
        with open(MERGED_DATA_DIR / "merged_test_labels.pkl", 'wb') as f:
            pickle.dump(self.merged_labels['test'], f)
        
        # Save label mapping
        with open(MERGED_DATA_DIR / "label_mapping.json", 'w') as f:
            json.dump(self.merged_label_map, f, indent=2)
        
        # Save metadata
        metadata = {
            "dataset_size": {
                "train": len(self.merged_data['train']),
                "test": len(self.merged_data['test']),
            },
            "skeleton_shape": (SKELETON_PARAMS["sequence_length"], SKELETON_PARAMS["num_joints"] * 3),
            "num_classes": len(self.merged_label_map),
            "pipeline_version": "1.0",
        }
        with open(MERGED_DATA_DIR / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Exported to {MERGED_DATA_DIR}")
        logger.info(f"   merged_train_data.pkl")
        logger.info(f"   merged_train_labels.pkl")
        logger.info(f"   merged_test_data.pkl")
        logger.info(f"   merged_test_labels.pkl")
        logger.info(f"   label_mapping.json")
        logger.info(f"   metadata.json")
    
    def run(self):
        """Run the complete pipeline"""
        logger.info("\n\n")
        logger.info("╔" + "="*68 + "╗")
        logger.info("║" + " "*15 + "DATA PIPELINE ORCHESTRATOR" + " "*27 + "║")
        logger.info("╚" + "="*68 + "╝")
        
        try:
            # Step 1: Ingest
            if not self.step_1_ingest_data():
                return False
            
            # Step 2: Extract Skating Verse skeletons
            skating_verse_skeletons, sv_valid_indices = self.step_2_extract_skating_verse_skeletons()
            
            # Step 3: Standardize
            train_std, test_std, sv_std = self.step_3_standardize_all_data(skating_verse_skeletons)
            
            # Step 4: Map labels
            train_labels, test_labels, sv_labels = self.step_4_map_labels(sv_valid_indices)
            
            # Step 5: Merge
            merged_train_data, merged_train_labels, merged_test_data, merged_test_labels = \
                self.step_5_merge_datasets(train_std, test_std, sv_std, 
                                          train_labels, test_labels, sv_labels)
            
            # Step 6: Balance & augment
            balanced_data, balanced_labels = self.step_6_balance_and_augment()
            
            # Step 7: Export
            self.step_7_export()
            
            logger.info("\n" + "="*70)
            logger.info("✅ PIPELINE COMPLETE!")
            logger.info("="*70 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


# Entry point
if __name__ == "__main__":
    pipeline = DataPipeline()
    success = pipeline.run()
    exit(0 if success else 1)
