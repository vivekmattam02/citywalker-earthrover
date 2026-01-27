"""
Visual Odometry for Indoor Position Tracking

Estimates robot position by tracking features between camera frames.

HOW IT WORKS:
1. Detect ORB features in consecutive frames
2. Match features between frames
3. Use Essential Matrix to estimate camera motion
4. Accumulate motion over time

LIMITATIONS:
- Scale ambiguity: Cannot determine absolute scale (we estimate it)
- Drift: Errors accumulate over time
- Needs texture: Fails in featureless environments

Author: Vivek Mattam
"""

import cv2
import numpy as np
from typing import Tuple, List
from orb_features import ORBFeatureExtractor


class VisualOdometry:
    """
    Tracks camera position using visual features.

    Returns 2D position (x, y) suitable for CityWalker coordinates.
    """

    def __init__(self, image_size: Tuple[int, int] = (640, 480), scale: float = 0.1):
        """
        Initialize Visual Odometry.

        Args:
            image_size: (width, height) of camera images
            scale: Scale factor to convert to meters (tune this for your robot)
        """
        self.image_size = image_size
        self.scale = scale

        # Camera intrinsic matrix (approximate for typical webcam)
        focal_length = image_size[0]
        cx, cy = image_size[0] / 2, image_size[1] / 2
        self.K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Feature extractor
        self.feature_extractor = ORBFeatureExtractor(n_features=1000)

        # Previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None

        # Current position (x, y) and heading (theta)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # radians

        # Position history (for CityWalker)
        self.position_history = []
        self.max_history = 10

        # Stats
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, float, float, int]:
        """
        Process a new frame and update position estimate.

        Args:
            frame: BGR camera frame

        Returns:
            success: Whether estimation succeeded
            x: Current x position (meters)
            y: Current y position (meters)
            n_matches: Number of feature matches
        """
        self.frame_count += 1

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect features
        keypoints, descriptors = self.feature_extractor.detect_and_compute(gray)

        if descriptors is None or len(keypoints) < 10:
            return False, self.x, self.y, 0

        # First frame - just store
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self._add_to_history()
            return True, self.x, self.y, 0

        # Match features
        matches = self.feature_extractor.match_features(
            self.prev_descriptors, descriptors, ratio_threshold=0.8
        )
        n_matches = len(matches)

        if n_matches < 8:
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return False, self.x, self.y, n_matches

        # Extract matched points
        pts1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints[m.trainIdx].pt for m in matches])

        # Estimate motion
        success, dx, dy, dtheta = self._estimate_motion(pts1, pts2)

        if success:
            # Update position (rotate by current heading)
            cos_t = np.cos(self.theta)
            sin_t = np.sin(self.theta)

            # Transform local motion to global frame
            global_dx = dx * cos_t - dy * sin_t
            global_dy = dx * sin_t + dy * cos_t

            self.x += global_dx * self.scale
            self.y += global_dy * self.scale
            self.theta += dtheta

            # Keep theta in [-pi, pi]
            while self.theta > np.pi:
                self.theta -= 2 * np.pi
            while self.theta < -np.pi:
                self.theta += 2 * np.pi

            self._add_to_history()

        # Update previous frame
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        return success, self.x, self.y, n_matches

    def _estimate_motion(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Tuple[bool, float, float, float]:
        """
        Estimate motion from point correspondences.

        Returns:
            success: Whether estimation succeeded
            dx: Forward motion (in image scale)
            dy: Lateral motion (in image scale)
            dtheta: Rotation (radians)
        """
        # Find Essential Matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None or np.sum(mask) < 8:
            return False, 0, 0, 0

        # Recover pose
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        # Extract translation (forward is Z, lateral is X)
        dx = t[2, 0]  # Forward
        dy = t[0, 0]  # Lateral

        # Extract rotation around Y axis (yaw)
        dtheta = np.arctan2(R[0, 2], R[2, 2])

        return True, dx, dy, dtheta

    def _add_to_history(self):
        """Add current position to history."""
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)

    def get_position(self) -> Tuple[float, float]:
        """Get current (x, y) position in meters."""
        return self.x, self.y

    def get_heading(self) -> float:
        """Get current heading in radians."""
        return self.theta

    def get_coordinates_for_citywalker(self, target_x: float, target_y: float) -> np.ndarray:
        """
        Get coordinates array formatted for CityWalker model.

        CityWalker expects coordinates in ROBOT FRAME:
            - X = forward (positive ahead of robot)
            - Y = left (positive to robot's left)

        VO World Frame (different from GPS world frame!):
            - Origin = robot's starting position
            - X-axis = robot's initial forward direction
            - Y-axis = robot's initial left direction
            - theta = accumulated rotation from initial heading

        Args:
            target_x, target_y: Target position in meters (VO world frame)

        Returns:
            coordinates: (6, 2) array - 5 past positions + 1 target, all in robot frame
        """
        coords = np.zeros((6, 2), dtype=np.float32)

        # Fill past positions (relative to current position)
        history = list(self.position_history)

        # Pad history if needed
        while len(history) < 5:
            if history:
                history.insert(0, history[0])
            else:
                history.append((0.0, 0.0))

        # Use last 5 positions
        history = history[-5:]

        # Rotate from VO world frame to current robot frame
        # theta = accumulated rotation since start (positive = turned left/counter-clockwise)
        # To get to robot frame, rotate by -theta
        cos_h = np.cos(self.theta)
        sin_h = np.sin(self.theta)

        # Convert past positions to robot frame
        for i, (hx, hy) in enumerate(history):
            # Relative position in VO world frame
            rel_x = hx - self.x
            rel_y = hy - self.y
            # Rotate by -theta to align with current robot heading
            # Standard 2D rotation: x' = x*cos(-θ) - y*sin(-θ) = x*cos(θ) + y*sin(θ)
            #                      y' = x*sin(-θ) + y*cos(-θ) = -x*sin(θ) + y*cos(θ)
            coords[i, 0] = rel_x * cos_h + rel_y * sin_h
            coords[i, 1] = -rel_x * sin_h + rel_y * cos_h

        # Target in robot frame
        rel_x = target_x - self.x
        rel_y = target_y - self.y
        coords[5, 0] = rel_x * cos_h + rel_y * sin_h
        coords[5, 1] = -rel_x * sin_h + rel_y * cos_h

        return coords

    def reset(self):
        """Reset odometry to origin."""
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.position_history = []
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.frame_count = 0


def test_coordinate_conversion():
    """Unit test for robot frame coordinate conversion."""
    print("\nTesting coordinate conversion...")
    print("-" * 40)

    vo = VisualOdometry()

    # Test case 1: Robot at origin, facing forward (theta=0)
    # Target at (1, 0) world -> should be (1, 0) robot (straight ahead)
    vo.x, vo.y, vo.theta = 0, 0, 0
    vo.position_history = [(0, 0)] * 5
    coords = vo.get_coordinates_for_citywalker(1.0, 0.0)
    target_robot = coords[5]
    assert abs(target_robot[0] - 1.0) < 0.01 and abs(target_robot[1] - 0.0) < 0.01, \
        f"Test 1 failed: expected (1, 0), got ({target_robot[0]:.2f}, {target_robot[1]:.2f})"
    print(f"  Test 1 PASS: Forward target -> robot frame ({target_robot[0]:.2f}, {target_robot[1]:.2f})")

    # Test case 2: Robot at origin, turned 90° left (theta=π/2)
    # Target at (1, 0) world -> should be (0, -1) robot (to the right)
    vo.x, vo.y, vo.theta = 0, 0, np.pi / 2
    coords = vo.get_coordinates_for_citywalker(1.0, 0.0)
    target_robot = coords[5]
    assert abs(target_robot[0] - 0.0) < 0.01 and abs(target_robot[1] - (-1.0)) < 0.01, \
        f"Test 2 failed: expected (0, -1), got ({target_robot[0]:.2f}, {target_robot[1]:.2f})"
    print(f"  Test 2 PASS: Right-side target -> robot frame ({target_robot[0]:.2f}, {target_robot[1]:.2f})")

    # Test case 3: Robot at origin, turned 90° right (theta=-π/2)
    # Target at (1, 0) world -> should be (0, 1) robot (to the left)
    vo.x, vo.y, vo.theta = 0, 0, -np.pi / 2
    coords = vo.get_coordinates_for_citywalker(1.0, 0.0)
    target_robot = coords[5]
    assert abs(target_robot[0] - 0.0) < 0.01 and abs(target_robot[1] - 1.0) < 0.01, \
        f"Test 3 failed: expected (0, 1), got ({target_robot[0]:.2f}, {target_robot[1]:.2f})"
    print(f"  Test 3 PASS: Left-side target -> robot frame ({target_robot[0]:.2f}, {target_robot[1]:.2f})")

    # Test case 4: Robot at (2, 0), facing forward, target at (3, 1)
    # Relative: (1, 1) -> robot frame (1, 1) - ahead and to the left
    vo.x, vo.y, vo.theta = 2, 0, 0
    coords = vo.get_coordinates_for_citywalker(3.0, 1.0)
    target_robot = coords[5]
    assert abs(target_robot[0] - 1.0) < 0.01 and abs(target_robot[1] - 1.0) < 0.01, \
        f"Test 4 failed: expected (1, 1), got ({target_robot[0]:.2f}, {target_robot[1]:.2f})"
    print(f"  Test 4 PASS: Offset target -> robot frame ({target_robot[0]:.2f}, {target_robot[1]:.2f})")

    print("-" * 40)
    print("All coordinate conversion tests PASSED!")


# Quick test
if __name__ == "__main__":
    print("Testing Visual Odometry...")
    print("=" * 50)

    # First run coordinate conversion unit tests
    test_coordinate_conversion()

    print("\n" + "=" * 50)
    vo = VisualOdometry()
    print(f"Initial position: ({vo.x:.3f}, {vo.y:.3f})")

    # Test with webcam if available
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("Webcam opened. Press 'q' to quit.")

        for i in range(100):
            ret, frame = cap.read()
            if not ret:
                break

            success, x, y, matches = vo.process_frame(frame)

            if i % 10 == 0:
                print(f"Frame {i}: pos=({x:.3f}, {y:.3f}), heading={np.degrees(vo.theta):.1f}deg, matches={matches}")

            # Display
            cv2.putText(frame, f"Pos: ({x:.2f}, {y:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Visual Odometry", frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("No webcam available. Testing with random frames...")
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            success, x, y, matches = vo.process_frame(frame)
            print(f"Frame {i}: success={success}, matches={matches}")

    print("\nVisual Odometry test complete!")
