"""
Skeleton Extraction from Skating Verse Videos
Supports MediaPipe (recommended) and OpenPose
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import pickle

try:
    import mediapipe as mp
except ImportError:
    print("⚠️  MediaPipe not installed. Install: pip install mediapipe")

from config import SKELETON_PARAMS, MERGED_DATA_DIR

logger = logging.getLogger(__name__)

class SkeletonExtractor:
    def __init__(self, extractor_type="mediapipe"):
        self.extractor_type = extractor_type
        self.sequence_length = SKELETON_PARAMS["sequence_length"]
        self.num_joints = SKELETON_PARAMS["num_joints"]
        self.confidence_threshold = SKELETON_PARAMS["confidence_threshold"]
        
        # COCO 17 keypoint indices in MediaPipe's 33-point model
        # COCO: [nose, L_eye, R_eye, L_ear, R_ear, L_shoulder, R_shoulder, 
        #        L_elbow, R_elbow, L_wrist, R_wrist, L_hip, R_hip, 
        #        L_knee, R_knee, L_ankle, R_ankle]
        self.coco_indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        
        if extractor_type == "mediapipe":
            self._init_mediapipe()
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Pose"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            min_detection_confidence=self.confidence_threshold,
            min_tracking_confidence=self.confidence_threshold,
        )
        logger.info("✅ MediaPipe Pose initialized (will extract COCO 17 keypoints)")
    
    def extract_skeleton_from_video(self, video_path, skip_frames=1):
        """
        Extract 17-joint COCO skeleton sequence from a video file
        (Downsampled from MediaPipe's 33-point model for compatibility with current dataset)
        
        Args:
            video_path: Path to video file
            skip_frames: Skip N frames (e.g., skip_frames=2 takes every 2nd frame)
        
        Returns:
            skeleton: (sequence_length, 51) numpy array
                      [x1,y1,conf1, x2,y2,conf2, ...] for 17 joints (17*3=51 features)
            success: bool
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"⚠️  Cannot open video: {video_path}")
                return None, False
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            skeleton_sequence = []
            frame_count = 0
            extracted_frames = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if requested
                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue
                
                # Extract skeleton
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)
                
                # Get joint coordinates (COCO 17 keypoints from MediaPipe 33 joints)
                if results.pose_landmarks:
                    joints = []
                    # Only extract COCO 17 keypoints
                    for idx in self.coco_indices:
                        landmark = results.pose_landmarks.landmark[idx]
                        joints.extend([landmark.x, landmark.y, landmark.visibility])
                    skeleton_sequence.append(np.array(joints))
                    extracted_frames += 1
                else:
                    # No pose detected - fill with zeros (will be handled in standardization)
                    skeleton_sequence.append(np.zeros(self.num_joints * 3))
                    extracted_frames += 1
                
                frame_count += 1
            
            cap.release()
            
            if len(skeleton_sequence) == 0:
                logger.warning(f"⚠️  No frames extracted: {video_path}")
                return None, False
            
            # Pad or truncate to sequence_length
            skeleton_array = np.array(skeleton_sequence)
            skeleton_array = self._resize_sequence(skeleton_array)
            
            logger.info(f"✅ {video_path.name}: {extracted_frames} frames → {self.sequence_length} frames")
            return skeleton_array, True
            
        except Exception as e:
            logger.error(f"❌ Error extracting {video_path}: {e}")
            return None, False
    
    def _resize_sequence(self, sequence):
        """
        Pad or truncate sequence to self.sequence_length
        
        Args:
            sequence: (N, 51) array [17 joints × 3 features (x,y,confidence)]
        
        Returns:
            resized: (sequence_length, 51) array
        """
        n_frames = len(sequence)
        
        if n_frames >= self.sequence_length:
            # Truncate: sample evenly across sequence
            indices = np.linspace(0, n_frames - 1, self.sequence_length, dtype=int)
            return sequence[indices]
        else:
            # Pad with repetition of last frame
            padded = np.zeros((self.sequence_length, sequence.shape[1]))
            padded[:n_frames] = sequence
            # Repeat last frame to fill padding
            if n_frames > 0:
                padded[n_frames:] = sequence[-1]
            return padded
    
    def batch_extract(self, video_paths, output_file=None):
        """
        Extract skeletons from multiple videos
        
        Args:
            video_paths: List of video file paths
            output_file: Save extracted skeletons to pickle file
        
        Returns:
            skeletons: List of (sequence_length, num_joints*3) arrays
            valid_indices: Indices of successfully extracted videos
        """
        skeletons = []
        valid_indices = []
        
        for i, video_path in enumerate(tqdm(video_paths, desc="Extracting skeletons")):
            skeleton, success = self.extract_skeleton_from_video(video_path)
            if success:
                skeletons.append(skeleton)
                valid_indices.append(i)
            else:
                skeletons.append(None)
        
        skeletons = np.array([s for s in skeletons if s is not None])
        
        logger.info(f"✅ Extracted {len(skeletons)}/{len(video_paths)} videos successfully")
        
        if output_file:
            with open(output_file, 'wb') as f:
                pickle.dump(skeletons, f)
            logger.info(f"💾 Saved to {output_file}")
        
        return skeletons, valid_indices


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    extractor = SkeletonExtractor(extractor_type="mediapipe")
    
    # Test with a single video (if available)
    test_video = "/path/to/sample_video.mp4"
    if Path(test_video).exists():
        skeleton, success = extractor.extract_skeleton_from_video(test_video)
        if success:
            print(f"✅ Skeleton shape: {skeleton.shape}")
    else:
        print("⚠️  Test video not found. Provide a video path to test extraction.")
