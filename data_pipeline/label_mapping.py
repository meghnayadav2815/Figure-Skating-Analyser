"""
Label Mapping: Convert Skating Verse labels to your 19-class schema
"""
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class LabelMapper:
    def __init__(self, label_mapping_dict):
        """
        Args:
            label_mapping_dict: Dict mapping Skating Verse labels → your class names
                               {"1Axel": "Axel", "2Axel": "Axel", ...}
        """
        self.label_mapping = label_mapping_dict
        self.unmapped_labels = set()
    
    def map_label(self, skating_verse_label):
        """
        Map a single Skating Verse label to your class
        
        Args:
            skating_verse_label: String label from Skating Verse
        
        Returns:
            class_name: String (your 19 class name) or None if unmapped
        """
        if skating_verse_label in self.label_mapping:
            return self.label_mapping[skating_verse_label]
        else:
            self.unmapped_labels.add(skating_verse_label)
            return None
    
    def batch_map_labels(self, skating_verse_labels):
        """
        Map multiple labels
        
        Args:
            skating_verse_labels: List of string labels from Skating Verse
        
        Returns:
            mapped_labels: List of mapped class names (None for unmapped)
        """
        mapped = []
        for label in skating_verse_labels:
            mapped.append(self.map_label(label))
        
        if self.unmapped_labels:
            logger.warning(f"⚠️  Found {len(self.unmapped_labels)} unmapped labels:")
            for ulabel in list(self.unmapped_labels)[:10]:
                logger.warning(f"   - {ulabel}")
            if len(self.unmapped_labels) > 10:
                logger.warning(f"   ... and {len(self.unmapped_labels) - 10} more")
        
        return mapped
    
    def get_mapping_summary(self):
        """Print summary of label mappings"""
        print("\n" + "="*60)
        print("LABEL MAPPING SUMMARY")
        print("="*60)
        
        # Group by target class
        inverse_map = {}
        for sv_label, your_label in self.label_mapping.items():
            if your_label not in inverse_map:
                inverse_map[your_label] = []
            inverse_map[your_label].append(sv_label)
        
        for your_label in sorted(inverse_map.keys()):
            sv_sources = inverse_map[your_label]
            print(f"\n{your_label}:")
            for sv_label in sv_sources[:3]:
                print(f"  ← {sv_label}")
            if len(sv_sources) > 3:
                print(f"  ← ... and {len(sv_sources) - 3} more variants")
        
        print("\n" + "="*60 + "\n")

# Example: How to use
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Your mapping
    example_mapping = {
        "1Axel": "Axel",
        "2Axel": "Axel",
        "3Axel": "Axel",
        "1Flip": "Flip",
        # ... etc
    }
    
    mapper = LabelMapper(example_mapping)
    
    # Test
    test_labels = ["1Axel", "2Flip", "Loop", "UnknownMove"]
    mapped = mapper.batch_map_labels(test_labels)
    print(f"Original: {test_labels}")
    print(f"Mapped:   {mapped}")
    
    mapper.get_mapping_summary()
