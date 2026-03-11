"""
Standardization: Normalize skeleton data from multiple sources
Ensures compatibility between current dataset and Skating Verse extractions
"""
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

from config import NORM_PARAMS, SKELETON_PARAMS

logger = logging.getLogger(__name__)

class DataStandardizer:
    def __init__(self):
        self.scaler_fitted = False
        self.scalers = {}  # Per-joint scalers for normalization
        self.num_joints = SKELETON_PARAMS["num_joints"]
    
    def fit_normalization(self, all_data):
        """
        Fit normalization parameters on all data (current + Skating Verse)
        
        Args:
            all_data: List of (sequence_length, num_joints*3) arrays
        """
        if not NORM_PARAMS["normalize"]:
            logger.info("⏭️  Normalization disabled")
            return
        
        # Flatten all data for fitting
        all_flat = np.concatenate([arr.reshape(-1, self.num_joints * 3) for arr in all_data])
        
        # Fit scaler per joint
        for joint_idx in range(self.num_joints):
            start_idx = joint_idx * 3
            end_idx = start_idx + 3
            
            scaler = StandardScaler()
            scaler.fit(all_flat[:, start_idx:end_idx])
            self.scalers[joint_idx] = scaler
        
        self.scaler_fitted = True
        logger.info(f"✅ Fitted normalization for {self.num_joints} joints")
    
    def standardize_sequence(self, skeleton_array):
        """
        Standardize a single skeleton sequence
        
        Args:
            skeleton_array: (sequence_length, num_joints*3) array
        
        Returns:
            standardized: (sequence_length, num_joints*3) array
        """
        if not NORM_PARAMS["normalize"]:
            return skeleton_array
        
        if not self.scaler_fitted:
            logger.warning("⚠️  Scalers not fitted. Fit first with fit_normalization()")
            return skeleton_array
        
        standardized = skeleton_array.copy()
        
        for joint_idx in range(self.num_joints):
            start_idx = joint_idx * 3
            end_idx = start_idx + 3
            
            # Extract x, y, confidence
            joint_data = standardized[:, start_idx:end_idx]
            
            # Normalize only x, y (not confidence)
            if NORM_PARAMS["mean_center"]:
                joint_data[:, :2] = joint_data[:, :2] - np.mean(joint_data[:, :2], axis=0)
            
            if NORM_PARAMS["std_normalize"] and self.scaler_fitted:
                scaler = self.scalers[joint_idx]
                joint_data = scaler.transform(joint_data)
            
            standardized[:, start_idx:end_idx] = joint_data
        
        return standardized
    
    def handle_missing_data(self, skeleton_array, threshold=0.5):
        """
        Handle frames with low confidence or missing data
        
        Args:
            skeleton_array: (sequence_length, num_joints*3) array
            threshold: Confidence threshold (0.0 - 1.0)
        
        Returns:
            cleaned: (sequence_length, num_joints*3) array with interpolated/smoothed data
        """
        cleaned = skeleton_array.copy()
        
        for joint_idx in range(self.num_joints):
            confidence_idx = joint_idx * 3 + 2
            confid_col = cleaned[:, confidence_idx]
            
            # Find low-confidence frames
            low_conf_mask = confid_col < threshold
            
            if np.any(low_conf_mask):
                # Interpolate missing x, y values
                x_col_idx = joint_idx * 3
                y_col_idx = joint_idx * 3 + 1
                
                # Simple linear interpolation
                x_col = cleaned[:, x_col_idx]
                y_col = cleaned[:, y_col_idx]
                good_frames = np.where(~low_conf_mask)[0]
                
                if len(good_frames) > 1:
                    x_col[:] = np.interp(np.arange(len(x_col)), good_frames, x_col[good_frames])
                    y_col[:] = np.interp(np.arange(len(y_col)), good_frames, y_col[good_frames])
                    cleaned[:, x_col_idx] = x_col
                    cleaned[:, y_col_idx] = y_col
                
                # Set low-confidence values to 0
                cleaned[low_conf_mask, confidence_idx] = 0.0
        
        return cleaned
    
    def smooth_sequence(self, skeleton_array, window_size=3):
        """
        Smooth skeleton sequence using moving average
        Helps reduce jitter from pose detection
        
        Args:
            skeleton_array: (sequence_length, num_joints*3) array
            window_size: Size of smoothing window
        
        Returns:
            smoothed: (sequence_length, num_joints*3) array
        """
        from scipy.ndimage import uniform_filter1d
        
        smoothed = skeleton_array.copy()
        for joint_idx in range(self.num_joints):
            start_idx = joint_idx * 3
            end_idx = start_idx + 2  # Only smooth x, y (not confidence)
            
            smoothed[:, start_idx:end_idx] = uniform_filter1d(
                skeleton_array[:, start_idx:end_idx],
                size=window_size,
                axis=0,
                mode='nearest'
            )
        
        return smoothed
    
    def validate_standardized_data(self, skeleton_array):
        """
        Validate standardized data for anomalies
        
        Args:
            skeleton_array: (sequence_length, num_joints*3) array
        
        Returns:
            valid: bool
            issues: List of detected issues
        """
        issues = []
        
        # Check shape
        expected_shape = (SKELETON_PARAMS["sequence_length"], SKELETON_PARAMS["num_joints"] * 3)
        if skeleton_array.shape != expected_shape:
            issues.append(f"Shape mismatch: expected {expected_shape}, got {skeleton_array.shape}")
        
        # Check for NaN
        if np.any(np.isnan(skeleton_array)):
            issues.append(f"Contains NaN values")
        
        # Check for extreme values (outliers)
        std = np.std(skeleton_array)
        if std > 10:  # Likely unnormalized
            issues.append(f"High std dev {std:.2f} - may not be normalized")
        
        # Check confidence scores
        confidence_cols = [i*3 + 2 for i in range(self.num_joints)]
        mean_confidence = np.mean(skeleton_array[:, confidence_cols])
        if mean_confidence < 0.3:
            issues.append(f"Low mean confidence: {mean_confidence:.2f}")
        
        valid = len(issues) == 0
        return valid, issues
    
    def batch_standardize(self, all_data):
        """
        Standardize all data samples
        
        Args:
            all_data: List of (sequence_length, num_joints*3) arrays
        
        Returns:
            standardized_data: List of standardized arrays
        """
        logger.info(f"Standardizing {len(all_data)} sequences...")
        
        standardized_data = []
        issues_count = 0
        
        for i, skeleton in enumerate(all_data):
            # Handle missing/low-confidence data
            skeleton = self.handle_missing_data(skeleton)
            
            # Smooth
            skeleton = self.smooth_sequence(skeleton, window_size=3)
            
            # Standardize
            skeleton = self.standardize_sequence(skeleton)
            
            # Validate
            valid, issues = self.validate_standardized_data(skeleton)
            if not valid:
                issues_count += 1
                if i < 5:  # Log first 5
                    logger.warning(f"   Sample {i}: {', '.join(issues)}")
            
            standardized_data.append(skeleton)
        
        if issues_count > 0:
            logger.warning(f"⚠️  {issues_count} samples have quality issues")
        
        logger.info(f"✅ Standardization complete")
        return standardized_data


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data for testing
    dummy_skeleton = np.random.rand(500, 51*3)
    
    standardizer = DataStandardizer()
    
    # Fit on multiple sequences
    all_data = [dummy_skeleton] * 5
    standardizer.fit_normalization(all_data)
    
    # Standardize
    standardized = standardizer.batch_standardize(all_data)
    print(f"✅ Standardized {len(standardized)} sequences")
