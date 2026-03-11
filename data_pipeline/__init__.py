"""
Data Pipeline for merging current skating dataset + Skating Verse videos

Pipeline stages:
1. Ingestion: Load current data + video list
2. Extraction: Extract skeletons from Skating Verse videos (MediaPipe)
3. Standardization: Normalize all data to common format
4. Label Mapping: Convert Skating Verse labels to your 19 classes
5. Merging: Combine datasets
6. Balancing: Oversample rare classes + augmentation
7. Export: Save merged dataset as pickles
"""

from .ingestion import DataIngestor
from .skeleton_extraction import SkeletonExtractor
from .standardization import DataStandardizer
from .label_mapping import LabelMapper
from .pipeline import DataPipeline

__version__ = "1.0"
__all__ = [
    "DataIngestor",
    "SkeletonExtractor",
    "DataStandardizer",
    "LabelMapper",
    "DataPipeline",
]
