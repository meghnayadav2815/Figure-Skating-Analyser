"""
Data Ingestion: Load current dataset + Skating Verse videos
"""
import pickle
import json
from pathlib import Path
import numpy as np
import logging

from config import (
    CURRENT_DATA_DIR, SKATING_VERSE_DIR, SKELETON_PARAMS, LABEL_MAPPING
)

logger = logging.getLogger(__name__)

class DataIngestor:
    def __init__(self):
        self.current_data = None
        self.current_labels = None
        self.label_mapping = None
        
    def load_current_dataset(self):
        """Load your existing pickle dataset"""
        try:
            train_data_path = CURRENT_DATA_DIR / "train_data.pkl"
            train_label_path = CURRENT_DATA_DIR / "train_label.pkl"
            test_data_path = CURRENT_DATA_DIR / "test_data.pkl"
            test_label_path = CURRENT_DATA_DIR / "test_label.pkl"
            
            with open(train_data_path, 'rb') as f:
                train_data = pickle.load(f)
            with open(train_label_path, 'rb') as f:
                train_labels = pickle.load(f)
            with open(test_data_path, 'rb') as f:
                test_data = pickle.load(f)
            with open(test_label_path, 'rb') as f:
                test_labels = pickle.load(f)
            
            self.current_data = {
                'train': train_data,
                'test': test_data,
            }
            self.current_labels = {
                'train': train_labels,
                'test': test_labels,
            }
            
            logger.info(f"📦 Loaded current dataset:")
            logger.info(f"   Train: {train_data.shape} samples, {train_labels.shape} labels")
            logger.info(f"   Test:  {test_data.shape} samples, {test_labels.shape} labels")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load current dataset: {e}")
            return False
    
    def load_label_mapping(self):
        """Load label mapping from your current data"""
        try:
            mapping_path = CURRENT_DATA_DIR / "label_mapping.json"
            with open(mapping_path, 'r') as f:
                self.label_mapping = json.load(f)
            
            logger.info(f"✅ Loaded label mapping with {len(self.label_mapping)} classes")
            return self.label_mapping
        except Exception as e:
            logger.error(f"❌ Failed to load label mapping: {e}")
            return None
    
    def list_skating_verse_videos(self):
        """List all Skating Verse video files"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        videos = []
        
        if SKATING_VERSE_DIR.exists():
            for video_file in SKATING_VERSE_DIR.rglob('*'):
                if video_file.suffix.lower() in video_extensions:
                    videos.append(str(video_file))
        
        logger.info(f"📹 Found {len(videos)} Skating Verse videos")
        return videos
    
    def validate_current_data_shapes(self):
        """Check if current data matches expected skeleton format"""
        if self.current_data is None:
            logger.warning("⚠️  Current data not loaded")
            return False
        
        train_shape = self.current_data['train'].shape
        expected_shape = (None, SKELETON_PARAMS['sequence_length'], SKELETON_PARAMS['num_joints'])
        
        if len(train_shape) == 3 and train_shape[1:] == expected_shape[1:]:
            logger.info(f"✅ Current data shape matches: {train_shape}")
            return True
        else:
            logger.error(f"❌ Shape mismatch! Expected (N, {SKELETON_PARAMS['sequence_length']}, {SKELETON_PARAMS['num_joints']}), got {train_shape}")
            return False
    
    def get_summary(self):
        """Print summary of loaded data"""
        print("\n" + "="*60)
        print("DATA INGESTION SUMMARY")
        print("="*60)
        
        if self.current_data:
            print(f"\n📦 CURRENT DATASET:")
            print(f"   Train samples: {self.current_data['train'].shape[0]}")
            print(f"   Test samples:  {self.current_data['test'].shape[0]}")
            print(f"   Skeleton shape: {SKELETON_PARAMS['sequence_length']} frames × {SKELETON_PARAMS['num_joints']} joints")
        
        if self.label_mapping:
            print(f"\n🏷️  LABEL MAPPING ({len(self.label_mapping)} classes):")
            for i, (idx, label) in enumerate(self.label_mapping.items()):
                if i < 5:  # Show first 5
                    print(f"   {idx}: {label}")
            if len(self.label_mapping) > 5:
                print(f"   ... and {len(self.label_mapping) - 5} more")
        
        videos = self.list_skating_verse_videos()
        print(f"\n📹 SKATING VERSE DATA:")
        print(f"   Videos to process: {len(videos)}")
        
        print("\n" + "="*60 + "\n")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    ingestor = DataIngestor()
    ingestor.load_current_dataset()
    ingestor.load_label_mapping()
    ingestor.get_summary()
