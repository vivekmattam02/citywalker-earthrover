"""
ORB Feature Extraction and Matching

ORB = Oriented FAST and Rotated BRIEF

Detects corners in images and creates descriptors for matching.

Author: Vivek Mattam
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class ORBFeatureExtractor:
    """Extracts and matches ORB features between images."""

    def __init__(self, n_features: int = 1000, scale_factor: float = 1.2, n_levels: int = 8,
                 use_ratio_test: bool = True):
        """
        Initialize ORB feature extractor.

        Args:
            n_features: Maximum number of features to detect
            scale_factor: Pyramid decimation ratio
            n_levels: Number of pyramid levels
            use_ratio_test: If True, use Lowe's ratio test for matching.
                           If False, use cross-check matching.
        """
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
        self.use_ratio_test = use_ratio_test

        if use_ratio_test:
            # For ratio test, we need knnMatch which requires crossCheck=False
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            # Cross-check matching (simpler but can't use ratio test)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
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
        """
        Match features between two sets of descriptors.

        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            ratio_threshold: Lowe's ratio test threshold (only used if use_ratio_test=True)
                           Lower values = stricter matching. Typical: 0.7-0.8

        Returns:
            List of good matches
        """
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) < 2 or len(desc2) < 2:
            return []

        if self.use_ratio_test:
            # Lowe's ratio test: keep match only if best match is significantly
            # better than second-best match
            knn_matches = self.matcher.knnMatch(desc1, desc2, k=2)

            good_matches = []
            for match_pair in knn_matches:
                # Some matches may have fewer than 2 neighbors
                if len(match_pair) < 2:
                    continue
                m, n = match_pair
                # Ratio test: m.distance should be much smaller than n.distance
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

            matches = sorted(good_matches, key=lambda x: x.distance)
        else:
            # Simple cross-check matching (ratio_threshold ignored)
            matches = self.matcher.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)

        return matches
