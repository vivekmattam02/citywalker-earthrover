#!/usr/bin/env python3
"""
CityWalker Navigation for FrodoBots EarthRover

Uses CityWalker vision model for GPS-based outdoor navigation.
Coordinate handling matches teleop_dataset.py from the CityWalker repo exactly.

Key design decisions (from CityWalker source code):
- Inference at 1Hz (model trained at target_fps=1)
- Heading derived from GPS trajectory, not compass
- step_scale computed from actual GPS movement
- Coordinates: translate to current origin, rotate so movement = +Y
- Control values in [-1, 1] range (EarthRover SDK)

Usage:
    python scripts/navigate_citywalker.py \
        --target-lat 40.7128 --target-lon -74.0060
"""

import sys
import os
import time
import argparse
import numpy as np
from collections import deque
import signal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from citywalker_wrapper import CityWalkerWrapper
from earthrover_interface import EarthRoverInterface


class CityWalkerNavigator:
    """
    CityWalker navigator for FrodoBots EarthRover.

    Coordinate transform matches CityWalker/data/teleop_dataset.py:
    1. Convert GPS (lat, lon) to local ENU meters (x=east, y=north)
    2. Store last 5 GPS positions
    3. Translate so current position = origin
    4. Rotate so movement direction = +Y axis (from GPS trajectory)
    5. Divide by step_scale (average distance between consecutive positions)

    Model was trained at 1Hz (target_fps=1), so each waypoint = 1 second ahead.
    """

    def __init__(
        self,
        target_lat,
        target_lon,
        inference_rate=1.0,         # 1Hz matches training target_fps=1
        control_rate=10.0,          # Send commands at 10Hz
        gps_filter_threshold=10.0,  # Reject GPS jumps > 10m
        goal_tolerance=5.0,         # 5m from paper
        waypoint_index=2,           # Middle waypoint (0-4)
        kp_linear=0.4,             # PD gain for forward speed
        kp_angular=0.6,            # PD gain for turning
        max_linear=0.5,            # Max linear command [-1, 1]
        max_angular=0.5,           # Max angular command [-1, 1]
        verbose=True
    ):
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.inference_rate = inference_rate
        self.control_rate = control_rate
        self.gps_filter_threshold = gps_filter_threshold
        self.goal_tolerance = goal_tolerance
        self.waypoint_index = waypoint_index
        self.kp_linear = kp_linear
        self.kp_angular = kp_angular
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.verbose = verbose

        # Timing
        self.inference_interval = 1.0 / inference_rate
        self.control_interval = 1.0 / control_rate

        # GPS reference point (set on first reading)
        self.ref_lat = None
        self.ref_lon = None

        # GPS history: last 5 positions in local ENU meters (x, y)
        self.gps_history = deque(maxlen=5)

        # Image buffer: last 5 camera frames
        self.image_buffer = deque(maxlen=5)

        # GPS filtering
        self.last_valid_gps = None
        self.gps_jump_count = 0

        # Navigation state
        self.running = False
        self.last_inference_time = 0
        self.step_scale = 1.0  # Will be computed from GPS trajectory

        # Print config
        print("\n" + "=" * 60)
        print("CITYWALKER NAVIGATION")
        print("=" * 60)
        print(f"Target: ({target_lat:.6f}, {target_lon:.6f})")
        print(f"Inference rate: {inference_rate} Hz (model trained at 1Hz)")
        print(f"Control rate: {control_rate} Hz")
        print(f"Goal tolerance: {goal_tolerance}m")
        print(f"Waypoint index: {waypoint_index}")
        print("=" * 60 + "\n")

        # Connect to robot
        print("[1/3] Connecting to robot...")
        self.rover = EarthRoverInterface()
        if not self.rover.connect():
            print("FAILED - Is SDK running?")
            print("  cd earth-rovers-sdk && hypercorn main:app --reload")
            sys.exit(1)

        # Load CityWalker
        print("[2/3] Loading CityWalker model...")
        self.model = CityWalkerWrapper()

        print("[3/3] Ready!\n")

        # Ctrl+C handler
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\n\nCTRL+C - stopping robot...")
        self.running = False
        self.rover.stop()
        print("Robot stopped.")
        sys.exit(0)

    # ---- GPS Utilities (matching teleop_dataset.py) ----

    def latlon_to_local(self, lat, lon):
        """
        Convert GPS to local ENU meters.
        Matches CityWalker/data/teleop_dataset.py latlon_to_local().

        Returns: (x, y) where x=east, y=north in meters
        """
        R_earth = 6378137  # meters (same as teleop_dataset.py)
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        lat0_rad = np.radians(self.ref_lat)
        lon0_rad = np.radians(self.ref_lon)
        dlat = lat_rad - lat0_rad
        dlon = lon_rad - lon0_rad
        x = dlon * np.cos((lat_rad + lat0_rad) / 2) * R_earth  # East
        y = dlat * R_earth                                        # North
        return np.array([x, y], dtype=np.float32)

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Distance in meters between two GPS points."""
        R = 6371000
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    def filter_gps(self, lat, lon):
        """Reject GPS jumps > threshold. Returns (lat, lon, accepted)."""
        if self.last_valid_gps is None:
            self.last_valid_gps = (lat, lon)
            return lat, lon, True

        dist = self.haversine_distance(self.last_valid_gps[0], self.last_valid_gps[1], lat, lon)
        if dist > self.gps_filter_threshold:
            self.gps_jump_count += 1
            if self.verbose:
                print(f"  GPS jump: {dist:.1f}m (rejected)")
            return self.last_valid_gps[0], self.last_valid_gps[1], False

        self.last_valid_gps = (lat, lon)
        return lat, lon, True

    # ---- Coordinate Transform (matching teleop_dataset.py) ----

    def transform_input(self, gps_positions):
        """
        Transform GPS positions to model's coordinate frame.
        Matches CityWalker/data/teleop_dataset.py transform_input().

        Args:
            gps_positions: numpy array (N, 2) of (x, y) in local meters

        Returns:
            rotated: numpy array (N, 2) in model's frame
            angle: rotation angle (needed to transform target too)
        """
        # Translate so current position = origin
        current = gps_positions[-1].copy()
        translated = gps_positions - current

        # Compute heading from GPS trajectory
        # Direction from second_last to current = movement direction
        second_last = translated[-2]

        # Rotate so movement direction aligns with +Y axis
        # This matches teleop_dataset.py exactly:
        # angle = -pi/2 - arctan2(second_last[1], second_last[0])
        angle = -np.pi / 2 - np.arctan2(second_last[1], second_last[0])

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])

        rotated = translated[:, :2] @ rotation_matrix.T
        return rotated, angle, current

    def compute_step_scale(self, gps_positions):
        """
        Compute step_scale from GPS trajectory.
        Matches CityWalker/data/teleop_dataset.py:
            step_scale = np.linalg.norm(np.diff(pose[:, [0, 1]], axis=0), axis=1).mean()

        Returns: float (average meters between consecutive positions)
        """
        if len(gps_positions) < 2:
            return 1.0  # Default if not enough data

        diffs = np.diff(np.array(gps_positions), axis=0)
        distances = np.linalg.norm(diffs, axis=1)

        # Filter out zero/tiny movements (robot stationary)
        distances = distances[distances > 0.01]

        if len(distances) == 0:
            return 1.0

        step_scale = distances.mean()
        return max(step_scale, 0.01)  # Clamp minimum

    def create_model_input(self, current_lat, current_lon):
        """
        Create the (6, 2) coordinate input for CityWalker.

        Returns:
            coords: numpy array (6, 2) - 5 past + 1 target, in model frame, meters
            step_scale: float - average step distance
            or None if not enough GPS history
        """
        if len(self.gps_history) < 2:
            return None, None

        # Pad GPS history to 5 if needed
        gps_array = np.array(list(self.gps_history), dtype=np.float32)
        while gps_array.shape[0] < 5:
            # Pad by repeating earliest position
            gps_array = np.vstack([gps_array[0:1], gps_array])

        # Use last 5 positions
        gps_5 = gps_array[-5:]

        # Transform: translate + rotate (matching teleop_dataset.py)
        transformed, angle, current_local = self.transform_input(gps_5)

        # Transform target GPS the same way
        target_local = self.latlon_to_local(self.target_lat, self.target_lon)
        target_translated = target_local - current_local

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        target_rotated = target_translated @ rotation_matrix.T

        # Build (6, 2) coordinate array
        coords = np.zeros((6, 2), dtype=np.float32)
        coords[:5] = transformed
        coords[5] = target_rotated

        # Compute step_scale from GPS trajectory
        step_scale = self.compute_step_scale(list(self.gps_history))

        return coords, step_scale

    # ---- Main Navigation ----

    def navigate(self):
        """Main navigation loop."""
        self.running = True
        step = 0
        last_control_time = time.time()
        current_waypoint = None

        print("STARTING NAVIGATION\n")

        while self.running:
            loop_start = time.time()
            step += 1

            # Get sensor data
            try:
                lat, lon, heading = self.rover.get_pose()
                if lat is None or lon is None:
                    time.sleep(0.1)
                    continue
            except Exception as e:
                print(f"Sensor error: {e}")
                time.sleep(0.1)
                continue

            # Filter GPS
            lat, lon, gps_ok = self.filter_gps(lat, lon)

            # Set reference point on first valid GPS
            if self.ref_lat is None:
                self.ref_lat = lat
                self.ref_lon = lon
                print(f"GPS reference set: ({lat:.6f}, {lon:.6f})")

            # Convert to local meters and store in history
            local_pos = self.latlon_to_local(lat, lon)
            self.gps_history.append(local_pos)

            # Check arrival
            dist_to_goal = self.haversine_distance(lat, lon, self.target_lat, self.target_lon)
            if dist_to_goal < self.goal_tolerance:
                print(f"\nGOAL REACHED! Distance: {dist_to_goal:.2f}m")
                print(f"Steps: {step}, GPS jumps rejected: {self.gps_jump_count}")
                self.rover.stop()
                break

            # Get camera frame
            try:
                frame = self.rover.get_camera_frame()
                if frame is not None:
                    self.image_buffer.append(frame)
            except Exception as e:
                print(f"Camera error: {e}")

            # Run CityWalker at inference rate
            now = time.time()
            if (len(self.image_buffer) >= 5 and
                len(self.gps_history) >= 2 and
                now - self.last_inference_time >= self.inference_interval):

                # Create coordinate input
                coords, step_scale = self.create_model_input(lat, lon)
                if coords is not None:
                    self.step_scale = step_scale

                    # Stack last 5 images
                    images = np.stack(list(self.image_buffer), axis=0)

                    try:
                        waypoints, arrival_prob = self.model.predict(
                            images, coords, step_scale=step_scale
                        )
                        current_waypoint = waypoints[self.waypoint_index]
                        self.last_inference_time = now

                        if self.verbose:
                            print(f"\n[{step}] dist={dist_to_goal:.1f}m | "
                                  f"wp=({current_waypoint[0]:+.2f}, {current_waypoint[1]:+.2f})m | "
                                  f"arr={arrival_prob:.1%} | "
                                  f"scale={step_scale:.3f}")

                    except Exception as e:
                        print(f"  CityWalker error: {e}")

            # Send control commands at control rate
            if current_waypoint is not None:
                control_now = time.time()
                if control_now - last_control_time >= self.control_interval:
                    wp_x, wp_y = current_waypoint

                    # wp_y = forward distance, wp_x = lateral offset
                    # Angle to waypoint: atan2(x, y) since x=right, y=forward
                    wp_angle = np.arctan2(wp_x, wp_y)

                    # PD control → [-1, 1] range for EarthRover
                    linear = self.kp_linear * np.clip(wp_y, -2.0, 2.0)
                    angular = self.kp_angular * wp_angle

                    linear = np.clip(linear, -self.max_linear, self.max_linear)
                    angular = np.clip(angular, -self.max_angular, self.max_angular)

                    try:
                        self.rover.send_control(linear, angular)
                    except Exception as e:
                        print(f"  Control error: {e}")

                    last_control_time = control_now

            # Maintain loop timing
            elapsed = time.time() - loop_start
            sleep_time = self.control_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.rover.stop()
        print("Navigation complete.\n")


def main():
    parser = argparse.ArgumentParser(
        description="CityWalker GPS Navigation for FrodoBots EarthRover"
    )

    parser.add_argument('--target-lat', type=float, required=True)
    parser.add_argument('--target-lon', type=float, required=True)
    parser.add_argument('--inference-rate', type=float, default=1.0,
                       help='CityWalker inference Hz (default: 1.0, matching training)')
    parser.add_argument('--control-rate', type=float, default=10.0,
                       help='Control command Hz (default: 10.0)')
    parser.add_argument('--goal-tolerance', type=float, default=5.0,
                       help='Arrival distance meters (default: 5.0)')
    parser.add_argument('--waypoint-index', type=int, default=2, choices=[0,1,2,3,4],
                       help='Which waypoint to follow (default: 2=middle)')
    parser.add_argument('--kp-linear', type=float, default=0.4,
                       help='Linear PD gain (default: 0.4)')
    parser.add_argument('--kp-angular', type=float, default=0.6,
                       help='Angular PD gain (default: 0.6)')
    parser.add_argument('--max-linear', type=float, default=0.5,
                       help='Max linear command (default: 0.5)')
    parser.add_argument('--max-angular', type=float, default=0.5,
                       help='Max angular command (default: 0.5)')
    parser.add_argument('--gps-threshold', type=float, default=10.0,
                       help='GPS jump rejection meters (default: 10.0)')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    nav = CityWalkerNavigator(
        target_lat=args.target_lat,
        target_lon=args.target_lon,
        inference_rate=args.inference_rate,
        control_rate=args.control_rate,
        gps_filter_threshold=args.gps_threshold,
        goal_tolerance=args.goal_tolerance,
        waypoint_index=args.waypoint_index,
        kp_linear=args.kp_linear,
        kp_angular=args.kp_angular,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        verbose=not args.quiet
    )

    try:
        nav.navigate()
    except Exception as e:
        print(f"\nNavigation failed: {e}")
        import traceback
        traceback.print_exc()
        nav.rover.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
