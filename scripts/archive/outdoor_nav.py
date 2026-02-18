"""
Outdoor Navigation using GPS

Navigate to GPS coordinates using CityWalker for obstacle avoidance.

DIFFERENCE FROM INDOOR:
    - Indoor: Uses Visual Odometry (camera-based position tracking)
    - Outdoor: Uses GPS for position tracking

USAGE:
    # Navigate to specific GPS coordinates
    python scripts/outdoor_nav.py --target-lat 40.7580 --target-lon -73.9855

    # Test without moving (dry run)
    python scripts/outdoor_nav.py --dry-run

    # Test with robot but show current GPS
    python scripts/outdoor_nav.py --show-gps

Author: Vivek Mattam
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from citywalker_wrapper import CityWalkerWrapper
from earthrover_interface import EarthRoverInterface
from pd_controller import PDController
from coordinate_utils import CoordinateTransformer, haversine_distance


class OutdoorNavigator:
    """
    GPS-based outdoor navigation using CityWalker + GPS.

    Uses GPS for position tracking instead of Visual Odometry.
    """

    def __init__(self, dry_run=False):
        print("=" * 60)
        print("OUTDOOR GPS NAVIGATION")
        print("=" * 60)

        self.dry_run = dry_run

        # Initialize components
        print("\n[1] Loading CityWalker model...")
        self.model = CityWalkerWrapper()

        if not dry_run:
            print("\n[2] Connecting to robot...")
            self.rover = EarthRoverInterface(timeout=30.0)  # Longer timeout for Puppeteer initialization
            if not self.rover.connect():
                raise RuntimeError("Failed to connect to robot. Is SDK server running?")
        else:
            print("\n[2] Dry run mode - no robot connection")
            self.rover = None

        print("\n[3] Initializing GPS coordinate transformer...")
        self.transformer = CoordinateTransformer(history_size=5)

        print("\n[4] Initializing controller...")
        self.controller = PDController(
            kp_linear=1.5,    # More aggressive for outdoor
            kp_angular=2.0,
            kd_angular=0.1,
            max_linear=0.7,   # Allow faster speeds
            max_angular=0.7
        )

        # Waypoint selection for control
        # CityWalker predicts 5 waypoints at 5Hz (0, 0.2s, 0.4s, 0.6s, 0.8s, 1.0s ahead)
        # Using a farther waypoint gives better lookahead for GPS-based navigation
        self.waypoint_index = 2  # Use 3rd waypoint (0.4s ahead, ~1-2m for walking speed)

        # GPS-specific adaptations for CityWalker
        self.waypoint_scale_factor = 3.0  # Scale CityWalker waypoints by 3x
        self.use_intermediate_goals = False  # DISABLED - causing target to jump around

        # NO GPS smoothing - CoordinateTransformer already handles history
        # Smoothing was creating artificial positions that confused the transformer

        # Image buffer for CityWalker (needs 5 frames)
        self.image_buffer = []

        # Navigation state
        self.target_lat = None
        self.target_lon = None
        self.goal_tolerance = 1.0  # meters - tighter for short distances

        # Stuck detection — DISABLED for outdoor GPS navigation
        # GPS jitter (±2-5m) causes too many false positives
        # If you want stuck detection, increase threshold to 100+ steps
        self.enable_stuck_detection = False
        self.stuck_threshold = 100  # ~10 seconds at 10Hz
        self.stuck_counter = 0
        self.last_distance = float('inf')
        self.min_progress = 1.0  # meters — must move at least this much to count as progress
        
        # Heading filter (GPS heading is very noisy)
        self.heading_history = []
        self.heading_filter_size = 10  # Average last 10 readings

        print("\nOutdoorNavigator ready!")

    def set_target(self, lat: float, lon: float):
        """Set navigation target as GPS coordinates."""
        self.target_lat = lat
        self.target_lon = lon
        print(f"\nTarget set: ({lat}, {lon})")

    def show_current_gps(self):
        """Display current GPS position from robot."""
        if self.rover is None:
            print("No robot connected")
            return

        print("\n" + "=" * 60)
        print("CURRENT GPS POSITION")
        print("=" * 60)

        try:
            lat, lon, heading_rad = self.rover.get_pose()  # Returns radians!
            heading_deg = np.degrees(heading_rad) if heading_rad else 0
            print(f"\nLatitude:  {lat}")
            print(f"Longitude: {lon}")
            print(f"Heading:   {heading_deg:.1f} degrees")
            print(f"\nGoogle Maps: https://maps.google.com/?q={lat},{lon}")
        except Exception as e:
            print(f"Error getting GPS: {e}")

    def get_distance_to_target(self, current_lat, current_lon) -> float:
        """Calculate distance from current position to target in meters."""
        return haversine_distance(
            current_lat, current_lon,
            self.target_lat, self.target_lon
        )

    def has_arrived(self, current_lat, current_lon) -> bool:
        """Check if robot has reached the target."""
        return self.get_distance_to_target(current_lat, current_lon) < self.goal_tolerance

    def is_stuck(self) -> bool:
        """Check if robot is stuck (not making progress)."""
        return self.stuck_counter > self.stuck_threshold

    def recover_from_stuck(self):
        """Try to recover when stuck."""
        print("\n[!] Robot stuck, attempting recovery...")

        if self.rover is None:
            return

        # Back up a bit
        print("    Backing up...")
        for _ in range(10):
            self.rover.send_control(-0.2, 0.0)
            time.sleep(0.1)

        # Turn randomly
        turn_dir = np.random.choice([-1, 1])
        print(f"    Turning {'left' if turn_dir > 0 else 'right'}...")
        for _ in range(15):
            self.rover.send_control(0.0, turn_dir * 0.4)
            time.sleep(0.1)

        self.rover.stop()
        self.stuck_counter = 0
        self.last_distance = float('inf')

        # Reset PD controller — stale derivative from before recovery would cause a spike
        self.controller.reset()

        # Clear image buffer — frames from before recovery are facing wrong direction
        self.image_buffer.clear()

        print("    Recovery complete, resuming navigation")

    def navigate_step(self) -> dict:
        """
        Execute one navigation step.

        Returns:
            status: Dict with navigation info
        """
        if self.rover is None:
            return {'success': False, 'reason': 'no_robot'}

        # Get camera frame
        frame = self.rover.get_camera_frame()
        if frame is None:
            self.rover.stop()  # Safety: don't run stale command
            return {'success': False, 'reason': 'no_frame'}

        # Get GPS position
        try:
            lat, lon, raw_heading_rad = self.rover.get_pose()  # get_pose already returns radians!

            if lat is None or lon is None:
                self.rover.stop()
                return {'success': False, 'reason': 'no_gps'}

            # Filter the heading (GPS heading is VERY noisy)
            if raw_heading_rad is not None:
                self.heading_history.append(raw_heading_rad)
                if len(self.heading_history) > self.heading_filter_size:
                    self.heading_history.pop(0)

                # Average the headings (handle wrap-around using sin/cos)
                sin_sum = sum(np.sin(h) for h in self.heading_history)
                cos_sum = sum(np.cos(h) for h in self.heading_history)
                heading_rad = np.arctan2(sin_sum, cos_sum)
            else:
                heading_rad = 0.0

            heading_deg = np.degrees(heading_rad)  # For display only
        except Exception as e:
            self.rover.stop()  # Safety: don't run stale command
            return {'success': False, 'reason': f'gps_error: {e}'}

        # Update coordinate transformer
        self.transformer.update(lat, lon, heading_rad)

        # Update image buffer
        self.image_buffer.append(frame)
        if len(self.image_buffer) > 5:
            self.image_buffer.pop(0)

        # Need 5 frames to run model
        if len(self.image_buffer) < 5:
            return {
                'success': True,
                'buffering': True,
                'buffer_size': len(self.image_buffer)
            }

        # Wait for coordinate transformer to be ready
        if not self.transformer.is_ready():
            return {
                'success': True,
                'buffering': True,
                'reason': 'building_gps_history'
            }

        # Check if arrived
        distance = self.get_distance_to_target(lat, lon)
        if self.has_arrived(lat, lon):
            self.rover.stop()
            return {
                'success': True,
                'arrived': True,
                'position': (lat, lon),
                'distance': distance
            }

        # Check for stuck condition (only if enabled)
        if self.enable_stuck_detection:
            if distance >= self.last_distance - self.min_progress:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            self.last_distance = distance

            if self.is_stuck():
                self.recover_from_stuck()
                return {
                    'success': True,
                    'recovering': True,
                    'position': (lat, lon)
                }

        # Get coordinates for CityWalker (in robot's local frame)
        coords = self.transformer.get_model_input(self.target_lat, self.target_lon)
        target_coord = coords[5]  # Last one is target (in robot frame)

        # Run CityWalker model inference
        try:
            images = np.stack(self.image_buffer, axis=0)
            waypoints, arrival_prob = self.model.predict(images, coords, step_scale=0.3)
        except Exception as e:
            self.rover.stop()
            return {'success': False, 'reason': f'model_error: {e}'}

        # SIMPLE APPROACH: Go directly to target, use CityWalker only for small corrections
        #
        # The target position keeps changing because GPS history is noisy.
        # CityWalker is predicting reasonable waypoints but they're too small.
        #
        # Solution: Go mostly toward target, blend in CityWalker for obstacle avoidance

        # Scaled CityWalker waypoint
        cw_wp = waypoints[self.waypoint_index] * self.waypoint_scale_factor

        # Target direction (from coords - in robot frame)
        target_wp = target_coord

        # MOSTLY use target direction, CityWalker adds small corrections
        # This keeps robot going toward goal while CityWalker handles obstacles
        if distance > 2.0:
            # Far: 90% target, 10% CityWalker
            wp = 0.9 * target_wp + 0.1 * cw_wp
        else:
            # Close: 70% target, 30% CityWalker
            wp = 0.7 * target_wp + 0.3 * cw_wp

        linear, angular = self.controller.compute(wp[0], wp[1])

        # Store for debugging
        citywalker_wp = cw_wp

        # Clip to safe outdoor speeds (no reverse)
        linear = np.clip(linear, 0.0, 0.7)
        angular = np.clip(angular, -0.7, 0.7)

        # Slow down if turning hard
        if abs(angular) > 0.4:
            linear *= 0.5

        # Ensure minimum forward speed so robot doesn't crawl
        if linear > 0.0:
            linear = max(linear, 0.15)

        # Send to robot
        self.rover.send_control(linear, angular)

        return {
            'success': True,
            'position': (lat, lon),
            'heading': heading_deg,
            'distance': distance,
            'target_local': (target_coord[0], target_coord[1]),
            'citywalker_wp': (citywalker_wp[0], citywalker_wp[1]),
            'waypoint': (wp[0], wp[1]),
            'velocity': (linear, angular),
            'arrival_prob': arrival_prob
        }

    def navigate_to_target(self, timeout: float = 300.0) -> bool:
        """
        Navigate to the current target.

        Args:
            timeout: Maximum time in seconds (default 5 minutes for outdoor)

        Returns:
            success: True if target reached
        """
        if self.target_lat is None or self.target_lon is None:
            print("Error: No target set!")
            return False

        print(f"\n{'='*60}")
        print(f"Navigating to GPS: ({self.target_lat}, {self.target_lon})")
        print(f"Timeout: {timeout:.0f} seconds")
        print(f"Goal tolerance: {self.goal_tolerance}m (GPS accuracy)")
        print(f"{'='*60}\n")

        start_time = time.time()
        step_count = 0

        try:
            while True:
                elapsed = time.time() - start_time

                # Check timeout
                if elapsed > timeout:
                    print(f"\n[!] Timeout reached ({timeout:.0f}s)")
                    if self.rover:
                        self.rover.stop()
                    return False

                # Execute navigation step
                status = self.navigate_step()
                step_count += 1

                # Handle buffering
                if status.get('buffering'):
                    reason = status.get('reason', f"{status.get('buffer_size', '?')}/5 frames")
                    print(f"\rBuffering... ({reason})", end="")
                    time.sleep(0.1)
                    continue

                # Check errors
                if not status.get('success'):
                    print(f"\nError: {status.get('reason')}")
                    time.sleep(0.5)
                    continue

                # Check if arrived
                if status.get('arrived'):
                    pos = status['position']
                    print(f"\n\n{'='*60}")
                    print("TARGET REACHED!")
                    print(f"Final position: ({pos[0]}, {pos[1]})")
                    print(f"Distance from target: {status['distance']:.1f}m")
                    print(f"Time taken: {elapsed:.1f} seconds")
                    print(f"{'='*60}")
                    return True

                # Print status every ~1 second (10 steps at 10Hz)
                if step_count % 10 == 0:
                    pos = status.get('position', (0, 0))
                    dist = status.get('distance', 0)
                    vel = status.get('velocity', (0, 0))
                    heading = status.get('heading', 0)
                    target = status.get('target_local', (0, 0))
                    wp = status.get('waypoint', (0, 0))
                    cw_wp = status.get('citywalker_wp', (0, 0))
                    arr = status.get('arrival_prob', 0)
                    print(f"\r[{elapsed:.0f}s] Dist: {dist:.1f}m | "
                          f"H: {heading:.0f}° | "
                          f"Tgt: ({target[0]:.1f}, {target[1]:.1f})m | "
                          f"CW: ({cw_wp[0]:.2f}, {cw_wp[1]:.2f})m | "
                          f"WP: ({wp[0]:.2f}, {wp[1]:.2f})m | "
                          f"V: ({vel[0]:.2f}, {vel[1]:.2f})   ", end="", flush=True)

                # 10Hz — model inference + GPS + frame fetch is heavy
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n[!] Navigation cancelled by user")
            if self.rover:
                self.rover.stop()
            return False

    def cleanup(self):
        """Stop robot and cleanup."""
        if self.rover:
            self.rover.stop()
        print("\nRobot stopped.")


def main():
    parser = argparse.ArgumentParser(description="Outdoor GPS Navigation")

    parser.add_argument(
        '--target-lat',
        type=float,
        help='Target latitude (e.g., 40.7580)'
    )
    parser.add_argument(
        '--target-lon',
        type=float,
        help='Target longitude (e.g., -73.9855)'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=300.0,
        help='Navigation timeout in seconds (default: 300)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test without robot connection'
    )
    parser.add_argument(
        '--show-gps',
        action='store_true',
        help='Just show current GPS position and exit'
    )

    args = parser.parse_args()

    # Show GPS mode
    if args.show_gps:
        rover = EarthRoverInterface(timeout=30.0)  # Longer timeout for Puppeteer initialization
        if rover.connect():
            lat, lon, heading_rad = rover.get_pose()  # Returns radians!
            heading_deg = np.degrees(heading_rad) if heading_rad else 0
            print("\n" + "=" * 60)
            print("CURRENT GPS POSITION")
            print("=" * 60)
            print(f"\nLatitude:  {lat}")
            print(f"Longitude: {lon}")
            print(f"Heading:   {heading_deg:.1f} degrees")
            print(f"\nGoogle Maps: https://maps.google.com/?q={lat},{lon}")
            print("\nUse this as reference to set your target!")
        return

    # Validate arguments
    if not args.dry_run and (args.target_lat is None or args.target_lon is None):
        parser.print_help()
        print("\n" + "=" * 60)
        print("HOW TO USE:")
        print("=" * 60)
        print("\n1. First, get your current GPS position:")
        print("   python scripts/outdoor_nav.py --show-gps")
        print("\n2. Pick a target location nearby")
        print("   (Use Google Maps to find GPS coordinates)")
        print("\n3. Run navigation:")
        print("   python scripts/outdoor_nav.py --target-lat 40.7580 --target-lon -73.9855")
        print("\n4. Or do a dry run to test:")
        print("   python scripts/outdoor_nav.py --dry-run")
        return

    # Create navigator
    try:
        nav = OutdoorNavigator(dry_run=args.dry_run)
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    try:
        if args.dry_run:
            print("\n" + "=" * 60)
            print("DRY RUN - Testing model inference")
            print("=" * 60)

            # Generate fake data
            fake_target_lat = 40.7580
            fake_target_lon = -73.9855
            nav.set_target(fake_target_lat, fake_target_lon)

            print("\nSimulating 5 frames...")
            for i in range(5):
                fake_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                nav.image_buffer.append(fake_image)

                # Simulate GPS update
                fake_lat = 40.7575 + i * 0.0001
                fake_lon = -73.9855
                nav.transformer.update(fake_lat, fake_lon, 0.0)

            print("Running model inference...")
            coords = nav.transformer.get_model_input(fake_target_lat, fake_target_lon)
            images = np.stack(nav.image_buffer, axis=0)

            start = time.time()
            waypoints, arrival_prob = nav.model.predict(images, coords, step_scale=0.3)
            inference_time = time.time() - start

            print(f"\nInference time: {inference_time*1000:.1f}ms")
            print(f"Waypoint 0: ({waypoints[0][0]:.3f}, {waypoints[0][1]:.3f})")
            print(f"Arrival prob: {arrival_prob:.3f}")
            print("\nDry run successful!")
        else:
            # Real navigation
            nav.set_target(args.target_lat, args.target_lon)
            nav.navigate_to_target(timeout=args.timeout)
    finally:
        nav.cleanup()


if __name__ == "__main__":
    main()
