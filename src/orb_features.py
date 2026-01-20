"""
ORB Feature Extraction and Matching

ORB = Oriented FAST and Rotated BRIEF

Detects corners in images and creates descriptors for matching.

Author: Vivek Mattam
"""

import cv2
import numpy as np
from typing import Tuple, List


class ORBFeatureExtractor:
    """Extracts and matches ORB features between images."""

    def __init__(self, n_features: int = 1000, scale_factor: float = 1.2, n_levels: int = 8):
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect keypoints and compute descriptors."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_threshold: float = 0.75
    ) -> List[cv2.DMatch]:
        """Match features between two sets of descriptors."""
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) < 2 or len(desc2) < 2:
            return []
        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
