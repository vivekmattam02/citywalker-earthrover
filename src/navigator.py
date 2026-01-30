"""
Navigator

Main control loop for autonomous navigation using CityWalker model.

This is the central piece that ties together:
- CityWalker model (waypoint prediction)
- Coordinate utils (GPS to local)
- PD controller (waypoints to velocity)
- EarthRover interface (robot control)

Usage:
    python src/navigator.py --target-lat 40.7580 --target-lon -73.9855
"""

import argparse
import time
import numpy as np
from collections import deque

from citywalker_wrapper import CityWalkerWrapper
from coordinate_utils import CoordinateTransformer
from pd_controller import PDController, StopController


class Navigator:
    """
    Autonomous navigation controller.

    Runs the main control loop: sense -> predict -> act.
    Optionally integrates depth safety to verify/override unsafe waypoints.
    """

    def __init__(
        self,
        rover_interface=None,
        arrival_threshold=2.0,
        arrival_prob_threshold=0.8,
        control_rate=10.0,
        step_length=0.1,
        waypoint_index=0,
        use_depth_safety=False,
        safety_margin=0.5,
        depth_model_size='small'
    ):
        """
        Initialize navigator.

        Args:
            rover_interface: EarthRover interface object (None for dry run)
            arrival_threshold: Distance in meters to consider arrived
            arrival_prob_threshold: Model confidence threshold for arrival
            control_rate: Control loop frequency in Hz
            step_length: Scale factor for waypoints (robot's step length)
            waypoint_index: Which predicted waypoint to target (0 = first)
            use_depth_safety: Enable runtime depth safety checking
            safety_margin: Safety margin in meters (for depth safety)
            depth_model_size: Depth model size ('small', 'base', 'large')
        """
        self.rover = rover_interface
        self.arrival_threshold = arrival_threshold
        self.arrival_prob_threshold = arrival_prob_threshold
        self.control_rate = control_rate
        self.dt = 1.0 / control_rate
        self.step_length = step_length
        self.waypoint_index = waypoint_index

        # Load model
        print("Loading CityWalker model...")
        self.model = CityWalkerWrapper()

        # Initialize coordinate transformer
        self.transformer = CoordinateTransformer(history_size=5)

        # Initialize controller
        self.controller = PDController(
            kp_linear=2.0,
            kd_linear=0.1,
            kp_angular=2.0,
            kd_angular=0.1
        )

        # Depth safety layer (optional)
        self.use_depth_safety = use_depth_safety
        if use_depth_safety:
            from depth_safety import DepthSafetyLayer
            print("Loading depth safety layer...")
            self.safety = DepthSafetyLayer(
                model_size=depth_model_size,
                margin=safety_margin
            )
        else:
            from depth_safety import DummyDepthSafety
            self.safety = DummyDepthSafety()

        # Image buffer (keep last 5 frames)
        self.context_size = self.model.get_context_size()
        self.image_buffer = deque(maxlen=self.context_size)

        # State
        self.target_lat = None
        self.target_lon = None
        self.running = False

        # Stats
        self.step_count = 0
        self.start_time = None

    def set_target(self, lat, lon):
        """Set the navigation target."""
        self.target_lat = lat
        self.target_lon = lon
        print(f"Target set: lat={lat}, lon={lon}")

    def update_state(self, image, lat, lon, heading):
        """
        Update robot state from sensors.

        Args:
            image: Camera frame (H, W, 3) numpy array, values 0-255
            lat: Current latitude
            lon: Current longitude
            heading: Current heading in radians
        """
        # Add image to buffer
        self.image_buffer.append(image)

        # Update coordinate transformer
        self.transformer.update(lat, lon, heading)

    def get_images(self):
        """Get stacked images for model input."""
        if len(self.image_buffer) < self.context_size:
            # Not enough frames yet, pad with copies of first frame
            if len(self.image_buffer) == 0:
                return None

            images = list(self.image_buffer)
            while len(images) < self.context_size:
                images.insert(0, images[0])  # Pad at beginning

            return np.stack(images, axis=0)

        return np.stack(list(self.image_buffer), axis=0)

    def is_ready(self):
        """Check if we have enough data to start navigation."""
        return (
            self.transformer.is_ready() and
            len(self.image_buffer) >= self.context_size and
            self.target_lat is not None
        )

    def check_arrival(self, arrival_prob):
        """Check if we've arrived at the target."""
        # Check distance
        distance = self.transformer.get_distance_to_target(
            self.target_lat, self.target_lon
        )

        if distance < self.arrival_threshold:
            print(f"Arrived! Distance: {distance:.1f}m")
            return True

        # Check model confidence
        if arrival_prob > self.arrival_prob_threshold:
            print(f"Model says arrived! Confidence: {arrival_prob:.2f}")
            return True

        return False

    def step(self):
        """
        Execute one control step.

        Returns:
            (linear_vel, angular_vel, waypoints, arrived)
        """
        if not self.is_ready():
            return 0.0, 0.0, None, False

        # Get model inputs
        images = self.get_images()
        coords = self.transformer.get_model_input(self.target_lat, self.target_lon)

        # Run model
        waypoints, arrival_prob = self.model.predict(
            images, coords, step_length=self.step_length
        )

        # Check if arrived
        if self.check_arrival(arrival_prob):
            return 0.0, 0.0, waypoints, True

        # Depth safety check (no-op if disabled)
        current_frame = self.image_buffer[-1]
        waypoints, was_overridden = self.safety.check_waypoints(
            current_frame, waypoints
        )

        # Compute velocity commands
        linear, angular = self.controller.compute_from_waypoints(
            waypoints, self.waypoint_index, self.dt
        )

        self.step_count += 1

        return linear, angular, waypoints, False

    def run(self, target_lat, target_lon, max_duration=300):
        """
        Run the navigation loop.

        Args:
            target_lat: Target latitude
            target_lon: Target longitude
            max_duration: Maximum run time in seconds

        Returns:
            True if arrived, False if timed out or stopped
        """
        if self.rover is None:
            print("Error: No rover interface provided")
            return False

        self.set_target(target_lat, target_lon)
        self.running = True
        self.start_time = time.time()
        self.step_count = 0

        print(f"Starting navigation to ({target_lat}, {target_lon})")
        print(f"Control rate: {self.control_rate} Hz")
        print("Waiting for sensor data...")

        try:
            while self.running:
                loop_start = time.time()

                # Check timeout
                elapsed = time.time() - self.start_time
                if elapsed > max_duration:
                    print(f"Timeout after {max_duration}s")
                    self.rover.stop()
                    return False

                # Get sensor data from rover
                image = self.rover.get_camera_frame()
                lat, lon, heading = self.rover.get_pose()

                # Update state
                self.update_state(image, lat, lon, heading)

                # Execute control step
                linear, angular, waypoints, arrived = self.step()

                if arrived:
                    self.rover.stop()
                    print(f"Navigation complete! Steps: {self.step_count}")
                    return True

                # Send command to rover
                self.rover.send_control(linear, angular)

                # Status update every second
                if self.step_count % int(self.control_rate) == 0:
                    distance = self.transformer.get_distance_to_target(
                        target_lat, target_lon
                    )
                    safety_stats = self.safety.get_stats()
                    safety_info = ""
                    if self.use_depth_safety:
                        safety_info = (f", overrides={safety_stats['total_overrides']}"
                                       f", clearance={safety_stats['forward_clearance'] or 0:.1f}m")
                    print(f"Step {self.step_count}: dist={distance:.1f}m, "
                          f"vel=({linear:.2f}, {angular:.2f}){safety_info}")

                # Maintain control rate
                loop_time = time.time() - loop_start
                sleep_time = self.dt - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopped by user")
            self.rover.stop()
            return False

        finally:
            self.running = False

    def stop(self):
        """Stop the navigation loop."""
        self.running = False
        if self.rover is not None:
            self.rover.stop()


# Dry run mode for testing without robot
class DummyRover:
    """Fake rover interface for testing."""

    def __init__(self):
        self.lat = 40.7580
        self.lon = -73.9855
        self.heading = 0.0
        self.step = 0

    def get_camera_frame(self):
        # Return random image
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def get_pose(self):
        # Simulate moving north
        self.step += 1
        self.lat += 0.000001 * self.step  # Move slightly north
        return self.lat, self.lon, self.heading

    def send_control(self, linear, angular):
        pass

    def stop(self):
        pass


def dry_run_test():
    """Test navigator without actual robot."""
    print("=" * 60)
    print("DRY RUN TEST (no robot)")
    print("=" * 60)

    # Create navigator with dummy rover
    dummy_rover = DummyRover()
    navigator = Navigator(rover_interface=dummy_rover)

    # Set target 100m north
    target_lat = 40.7590
    target_lon = -73.9855
    navigator.set_target(target_lat, target_lon)

    print("\nSimulating sensor updates...")

    # Simulate 10 control steps
    for i in range(10):
        # Get dummy sensor data
        image = dummy_rover.get_camera_frame()
        lat, lon, heading = dummy_rover.get_pose()

        # Update navigator state
        navigator.update_state(image, lat, lon, heading)

        # Check if ready
        if not navigator.is_ready():
            print(f"Step {i+1}: Warming up... (images: {len(navigator.image_buffer)}, "
                  f"coords ready: {navigator.transformer.is_ready()})")
            continue

        # Run one step
        linear, angular, waypoints, arrived = navigator.step()

        distance = navigator.transformer.get_distance_to_target(target_lat, target_lon)
        print(f"Step {i+1}: dist={distance:.1f}m, linear={linear:.3f}, angular={angular:.3f}")

        if waypoints is not None:
            print(f"  Waypoint 0: x={waypoints[0][0]:.3f}m, y={waypoints[0][1]:.3f}m")

    print("\nDry run complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CityWalker Navigation")
    parser.add_argument("--target-lat", type=float, help="Target latitude")
    parser.add_argument("--target-lon", type=float, help="Target longitude")
    parser.add_argument("--dry-run", action="store_true", help="Run without robot")
    parser.add_argument("--depth-safety", action="store_true",
                        help="Enable runtime depth safety checking")
    parser.add_argument("--safety-margin", type=float, default=0.5,
                        help="Safety margin in meters (default 0.5)")

    args = parser.parse_args()

    if args.dry_run or (args.target_lat is None and args.target_lon is None):
        dry_run_test()
    else:
        from earthrover_interface import EarthRoverInterface
        rover = EarthRoverInterface()
        if not rover.connect():
            print("Cannot connect to robot. Use --dry-run for testing.")
            exit(1)
        nav = Navigator(
            rover_interface=rover,
            use_depth_safety=args.depth_safety,
            safety_margin=args.safety_margin
        )
        nav.run(args.target_lat, args.target_lon)
