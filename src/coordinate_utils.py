"""
Coordinate Utilities

Handles GPS to local coordinate conversion and trajectory management.

Usage:
    from coordinate_utils import CoordinateTransformer

    transformer = CoordinateTransformer()
    transformer.update(gps_lat, gps_lon, heading)
    coords = transformer.get_model_input(target_lat, target_lon)
"""

import numpy as np
from collections import deque
import math


# Earth's radius in meters
EARTH_RADIUS = 6371000


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two GPS points in meters.

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)

    Returns:
        Distance in meters
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS * c


def gps_to_local(origin_lat, origin_lon, point_lat, point_lon):
    """
    Convert GPS coordinates to local (x, y) in meters.

    Origin is at (0, 0). X is East, Y is North.

    Args:
        origin_lat, origin_lon: Reference point (degrees)
        point_lat, point_lon: Point to convert (degrees)

    Returns:
        (x, y) in meters relative to origin
    """
    # X: East-West distance (longitude)
    x = haversine_distance(origin_lat, origin_lon, origin_lat, point_lon)
    if point_lon < origin_lon:
        x = -x

    # Y: North-South distance (latitude)
    y = haversine_distance(origin_lat, origin_lon, point_lat, origin_lon)
    if point_lat < origin_lat:
        y = -y

    return x, y


def rotate_to_robot_frame(x, y, heading):
    """
    Rotate coordinates from world frame to robot frame.

    World frame (GPS/map convention):
        - X = East (positive toward East)
        - Y = North (positive toward North)

    Robot frame:
        - X = Forward (positive ahead of robot)
        - Y = Left (positive to robot's left)

    Heading convention (compass style):
        - 0 = North (robot facing North)
        - π/2 = East (robot facing East)
        - π = South
        - Positive = clockwise rotation

    Args:
        x, y: Coordinates in world frame (X=East, Y=North) in meters
        heading: Robot heading in radians (0=North, positive=clockwise)

    Returns:
        (x_robot, y_robot) in robot's local frame (X=forward, Y=left) in meters

    Example:
        Robot facing North (heading=0), point 1m East of robot:
        >>> rotate_to_robot_frame(1, 0, 0)
        (0, -1)  # Point is to the right (negative Y in robot frame)

        Robot facing East (heading=π/2), point 1m East of robot:
        >>> rotate_to_robot_frame(1, 0, math.pi/2)
        (1, 0)  # Point is directly ahead
    """
    # Robot's forward direction in world frame: (sin(heading), cos(heading))
    # Robot's left direction in world frame: (-cos(heading), sin(heading))
    #
    # To convert world point to robot frame:
    # x_robot = dot((x, y), forward) = x*sin(h) + y*cos(h)
    # y_robot = dot((x, y), left) = -x*cos(h) + y*sin(h)

    sin_h = math.sin(heading)
    cos_h = math.cos(heading)

    x_robot = x * sin_h + y * cos_h
    y_robot = -x * cos_h + y * sin_h

    return x_robot, y_robot


class CoordinateTransformer:
    """
    Manages coordinate transformations and trajectory history.

    Keeps track of past positions and provides input for the CityWalker model.
    """

    def __init__(self, history_size=5):
        """
        Args:
            history_size: Number of past positions to keep (default 5 for CityWalker)
        """
        self.history_size = history_size

        # Current state
        self.current_lat = None
        self.current_lon = None
        self.current_heading = None

        # History of positions in world frame (lat, lon)
        self.gps_history = deque(maxlen=history_size)

        # Flag to check if we have enough history
        self.initialized = False

    def update(self, lat, lon, heading):
        """
        Update current position and heading.

        Call this every time you get new GPS/IMU data.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            heading: Heading in radians (0 = North, positive = clockwise)
        """
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading

        # Add to history
        self.gps_history.append((lat, lon))

        # Check if we have enough history
        if len(self.gps_history) >= self.history_size:
            self.initialized = True

    def get_trajectory_local(self):
        """
        Get past trajectory in robot's local frame.

        Returns:
            numpy array of shape (history_size, 2) with (x, y) positions.
            Most recent position is at index -1.
            Current robot position is (0, 0).
        """
        if not self.initialized:
            # Not enough history, return zeros
            return np.zeros((self.history_size, 2), dtype=np.float32)

        trajectory = []

        for lat, lon in self.gps_history:
            # Convert to local coordinates relative to current position
            x, y = gps_to_local(self.current_lat, self.current_lon, lat, lon)

            # Rotate to robot frame
            x_robot, y_robot = rotate_to_robot_frame(x, y, self.current_heading)

            trajectory.append([x_robot, y_robot])

        return np.array(trajectory, dtype=np.float32)

    def get_target_local(self, target_lat, target_lon):
        """
        Get target position in robot's local frame.

        Args:
            target_lat: Target latitude in degrees
            target_lon: Target longitude in degrees

        Returns:
            numpy array of shape (2,) with (x, y) position
        """
        if self.current_lat is None:
            return np.zeros(2, dtype=np.float32)

        # Convert to local coordinates
        x, y = gps_to_local(self.current_lat, self.current_lon, target_lat, target_lon)

        # Rotate to robot frame
        x_robot, y_robot = rotate_to_robot_frame(x, y, self.current_heading)

        return np.array([x_robot, y_robot], dtype=np.float32)

    def get_model_input(self, target_lat, target_lon):
        """
        Get the coordinate input for CityWalker model.

        Returns:
            numpy array of shape (6, 2):
                - Positions 0-4: Past trajectory (5 positions)
                - Position 5: Target location
        """
        trajectory = self.get_trajectory_local()  # (5, 2)
        target = self.get_target_local(target_lat, target_lon)  # (2,)

        # Combine: 5 past positions + 1 target
        coords = np.vstack([trajectory, target.reshape(1, 2)])  # (6, 2)

        return coords.astype(np.float32)

    def get_distance_to_target(self, target_lat, target_lon):
        """
        Get distance to target in meters.

        Args:
            target_lat: Target latitude
            target_lon: Target longitude

        Returns:
            Distance in meters
        """
        if self.current_lat is None:
            return float('inf')

        return haversine_distance(
            self.current_lat, self.current_lon,
            target_lat, target_lon
        )

    def is_ready(self):
        """Check if we have enough history to run the model."""
        return self.initialized


def test_rotate_to_robot_frame():
    """Unit test for rotate_to_robot_frame function."""
    print("\nTesting rotate_to_robot_frame...")
    print("-" * 40)

    # Test case 1: Robot facing North (heading=0), point 1m East
    # Should be to the right of robot -> (0, -1) in robot frame
    x_r, y_r = rotate_to_robot_frame(1, 0, 0)
    assert abs(x_r - 0) < 0.01 and abs(y_r - (-1)) < 0.01, \
        f"Test 1 failed: expected (0, -1), got ({x_r:.2f}, {y_r:.2f})"
    print(f"  Test 1 PASS: Facing North, point East -> ({x_r:.2f}, {y_r:.2f})")

    # Test case 2: Robot facing North (heading=0), point 1m North
    # Should be ahead of robot -> (1, 0) in robot frame
    x_r, y_r = rotate_to_robot_frame(0, 1, 0)
    assert abs(x_r - 1) < 0.01 and abs(y_r - 0) < 0.01, \
        f"Test 2 failed: expected (1, 0), got ({x_r:.2f}, {y_r:.2f})"
    print(f"  Test 2 PASS: Facing North, point North -> ({x_r:.2f}, {y_r:.2f})")

    # Test case 3: Robot facing East (heading=π/2), point 1m East
    # Should be ahead of robot -> (1, 0) in robot frame
    x_r, y_r = rotate_to_robot_frame(1, 0, math.pi/2)
    assert abs(x_r - 1) < 0.01 and abs(y_r - 0) < 0.01, \
        f"Test 3 failed: expected (1, 0), got ({x_r:.2f}, {y_r:.2f})"
    print(f"  Test 3 PASS: Facing East, point East -> ({x_r:.2f}, {y_r:.2f})")

    # Test case 4: Robot facing East (heading=π/2), point 1m North
    # Should be to the left of robot -> (0, 1) in robot frame
    x_r, y_r = rotate_to_robot_frame(0, 1, math.pi/2)
    assert abs(x_r - 0) < 0.01 and abs(y_r - 1) < 0.01, \
        f"Test 4 failed: expected (0, 1), got ({x_r:.2f}, {y_r:.2f})"
    print(f"  Test 4 PASS: Facing East, point North -> ({x_r:.2f}, {y_r:.2f})")

    # Test case 5: Robot facing South (heading=π), point 1m North
    # Should be behind robot -> (-1, 0) in robot frame
    x_r, y_r = rotate_to_robot_frame(0, 1, math.pi)
    assert abs(x_r - (-1)) < 0.01 and abs(y_r - 0) < 0.01, \
        f"Test 5 failed: expected (-1, 0), got ({x_r:.2f}, {y_r:.2f})"
    print(f"  Test 5 PASS: Facing South, point North -> ({x_r:.2f}, {y_r:.2f})")

    print("-" * 40)
    print("All rotate_to_robot_frame tests PASSED!")


# Quick test if run directly
if __name__ == "__main__":
    print("Testing CoordinateTransformer...")

    # First run unit tests
    test_rotate_to_robot_frame()

    print("\n" + "=" * 50)

    # Create transformer
    transformer = CoordinateTransformer(history_size=5)

    # Simulate robot moving north from Times Square
    start_lat = 40.7580
    start_lon = -73.9855

    # Simulate 6 position updates (moving north)
    for i in range(6):
        lat = start_lat + i * 0.0001  # ~11 meters north each step
        lon = start_lon
        heading = 0.0  # Facing north

        transformer.update(lat, lon, heading)
        print(f"Update {i+1}: lat={lat:.4f}, ready={transformer.is_ready()}")

    # Set a target 50 meters ahead (north of robot, which is facing north)
    target_lat = start_lat + 0.0005
    target_lon = start_lon

    # Get model input
    coords = transformer.get_model_input(target_lat, target_lon)
    distance = transformer.get_distance_to_target(target_lat, target_lon)

    print(f"\nModel input shape: {coords.shape}")
    print(f"Distance to target: {distance:.1f} meters")
    print(f"\nCoordinates (robot frame):")
    for i, (x, y) in enumerate(coords):
        label = f"Past {i+1}" if i < 5 else "Target"
        print(f"  {label}: x={x:.2f}m, y={y:.2f}m")

    # Verify target is ahead (positive X)
    target_x, target_y = coords[5]
    assert target_x > 0, f"Target should be ahead (positive X), got x={target_x:.2f}"
    assert abs(target_y) < 1, f"Target should be roughly centered, got y={target_y:.2f}"
    print(f"\nVerification: Target is ahead and centered. PASS!")

    print("\nCoordinate utils test passed!")
