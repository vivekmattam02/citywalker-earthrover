"""
Fully Autonomous Indoor Navigation

Give the robot a target position, it navigates there on its own.

HOW IT WORKS:
1. You specify a target (x, y) in meters
2. Visual Odometry tracks robot position
3. CityWalker handles obstacle avoidance
4. Robot autonomously navigates to target

USAGE:
    # Go to position (2, 0) - 2 meters forward
    python scripts/autonomous_nav.py --target 2.0 0.0

    # Go to position (1, 1) - 1m forward, 1m left
    python scripts/autonomous_nav.py --target 1.0 1.0

    # Explore autonomously (random targets)
    python scripts/autonomous_nav.py --explore

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
from visual_odometry import VisualOdometry


class AutonomousNavigator:
    """Fully autonomous indoor navigation using CityWalker + Visual Odometry."""

    def __init__(self):
        print("=" * 60)
        print("AUTONOMOUS INDOOR NAVIGATION")
        print("=" * 60)

        # Initialize components
        print("\n[1] Loading CityWalker model...")
        self.model = CityWalkerWrapper()

        print("\n[2] Connecting to robot...")
        self.rover = EarthRoverInterface()
        if not self.rover.connect():
            raise RuntimeError("Failed to connect to robot. Is SDK server running?")

        print("\n[3] Initializing Visual Odometry...")
        self.vo = VisualOdometry(image_size=(640, 480), scale=0.05)

        print("\n[4] Initializing controller...")
        self.controller = PDController()

        # Image buffer for CityWalker (needs 5 frames)
        self.image_buffer = []

        # Navigation state
        self.target_x = 0.0
        self.target_y = 0.0
        self.goal_tolerance = 0.2  # meters - how close is "arrived"
        self.stuck_threshold = 30  # frames without progress = stuck
        self.stuck_counter = 0
        self.last_distance = float('inf')

        print("\nAutonomousNavigator ready!")

    def set_target(self, x: float, y: float):
        """Set navigation target in meters."""
        self.target_x = x
        self.target_y = y
        print(f"\nTarget set: ({x:.2f}, {y:.2f}) meters")

    def get_distance_to_target(self) -> float:
        """Calculate distance from current position to target."""
        dx = self.target_x - self.vo.x
        dy = self.target_y - self.vo.y
        return np.sqrt(dx*dx + dy*dy)

    def has_arrived(self) -> bool:
        """Check if robot has reached the target."""
        return self.get_distance_to_target() < self.goal_tolerance

    def is_stuck(self) -> bool:
        """Check if robot is stuck (not making progress)."""
        return self.stuck_counter > self.stuck_threshold

    def recover_from_stuck(self):
        """Try to recover when stuck."""
        print("\n[!] Robot stuck, attempting recovery...")

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
        print("    Recovery complete, resuming navigation")

    def navigate_step(self) -> dict:
        """
        Execute one navigation step.

        Returns:
            status: Dict with navigation info
        """
        # Get camera frame
        frame = self.rover.get_camera_frame()
        if frame is None:
            return {'success': False, 'reason': 'no_frame'}

        # Update visual odometry
        vo_success, pos_x, pos_y, n_matches = self.vo.process_frame(frame)
        heading = self.vo.get_heading()

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

        # Check if arrived
        distance = self.get_distance_to_target()
        if self.has_arrived():
            self.rover.stop()
            return {
                'success': True,
                'arrived': True,
                'position': (pos_x, pos_y),
                'distance': distance
            }

        # Check for stuck condition
        if distance >= self.last_distance - 0.01:  # Not making progress
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_distance = distance

        if self.is_stuck():
            self.recover_from_stuck()
            return {
                'success': True,
                'recovering': True,
                'position': (pos_x, pos_y)
            }

        # Get coordinates for CityWalker
        coords = self.vo.get_coordinates_for_citywalker(self.target_x, self.target_y)

        # Run model
        images = np.stack(self.image_buffer, axis=0)
        waypoints, arrival_prob = self.model.predict(images, coords, step_scale=0.3)

        # Get first waypoint
        wp = waypoints[0]

        # Compute motor commands (don't reset controller - need derivative history for D term)
        linear, angular = self.controller.compute(wp[0], wp[1])

        # Safety scaling
        safe_linear = np.clip(linear * 0.6, 0.1, 0.3)  # 10-30% speed
        safe_angular = np.clip(angular * 0.5, -0.4, 0.4)

        # Send to robot
        self.rover.send_control(safe_linear, safe_angular)

        return {
            'success': True,
            'position': (pos_x, pos_y),
            'heading': np.degrees(heading),
            'distance': distance,
            'waypoint': (wp[0], wp[1]),
            'velocity': (safe_linear, safe_angular),
            'matches': n_matches,
            'arrival_prob': arrival_prob
        }

    def navigate_to_target(self, timeout: float = 120.0) -> bool:
        """
        Navigate to the current target.

        Args:
            timeout: Maximum time in seconds

        Returns:
            success: True if target reached
        """
        print(f"\n{'='*60}")
        print(f"Navigating to ({self.target_x:.2f}, {self.target_y:.2f})")
        print(f"Timeout: {timeout:.0f} seconds")
        print(f"{'='*60}\n")

        start_time = time.time()
        step_count = 0

        try:
            while True:
                elapsed = time.time() - start_time

                # Check timeout
                if elapsed > timeout:
                    print(f"\n[!] Timeout reached ({timeout:.0f}s)")
                    self.rover.stop()
                    return False

                # Execute navigation step
                status = self.navigate_step()
                step_count += 1

                # Handle buffering
                if status.get('buffering'):
                    print(f"\rBuffering... ({status['buffer_size']}/5 frames)", end="")
                    time.sleep(0.1)
                    continue

                # Check if arrived
                if status.get('arrived'):
                    print(f"\n\n{'='*60}")
                    print("TARGET REACHED!")
                    print(f"Final position: ({status['position'][0]:.2f}, {status['position'][1]:.2f})")
                    print(f"Time taken: {elapsed:.1f} seconds")
                    print(f"{'='*60}")
                    return True

                # Print status
                if step_count % 5 == 0:  # Every 5 steps
                    pos = status.get('position', (0, 0))
                    dist = status.get('distance', 0)
                    vel = status.get('velocity', (0, 0))
                    print(f"\rPos: ({pos[0]:+.2f}, {pos[1]:+.2f})  "
                          f"Dist: {dist:.2f}m  "
                          f"Vel: ({vel[0]:.2f}, {vel[1]:.2f})  "
                          f"Time: {elapsed:.0f}s   ", end="", flush=True)

                # Small delay
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n\n[!] Navigation cancelled by user")
            self.rover.stop()
            return False

    def explore(self, num_targets: int = 5, area_size: float = 2.0):
        """
        Autonomous exploration - navigate to random targets.

        Args:
            num_targets: Number of random targets to visit
            area_size: Size of exploration area in meters
        """
        print(f"\n{'='*60}")
        print(f"AUTONOMOUS EXPLORATION MODE")
        print(f"Visiting {num_targets} random targets in {area_size}m x {area_size}m area")
        print(f"{'='*60}")

        targets_reached = 0

        for i in range(num_targets):
            # Generate random target
            target_x = np.random.uniform(0.5, area_size)
            target_y = np.random.uniform(-area_size/2, area_size/2)

            print(f"\n[Target {i+1}/{num_targets}]")
            self.set_target(target_x, target_y)

            # Navigate to target
            if self.navigate_to_target(timeout=60.0):
                targets_reached += 1
                print(f"Progress: {targets_reached}/{i+1} targets reached")

                # Pause briefly at target
                time.sleep(1.0)
            else:
                print(f"Failed to reach target {i+1}")

            # Reset for next target
            self.vo.reset()
            self.stuck_counter = 0
            self.last_distance = float('inf')

        print(f"\n{'='*60}")
        print(f"EXPLORATION COMPLETE")
        print(f"Reached {targets_reached}/{num_targets} targets")
        print(f"{'='*60}")

    def cleanup(self):
        """Stop robot and cleanup."""
        self.rover.stop()
        print("\nRobot stopped.")


def main():
    parser = argparse.ArgumentParser(description="Autonomous Indoor Navigation")

    parser.add_argument(
        '--target',
        type=float,
        nargs=2,
        metavar=('X', 'Y'),
        help='Target position in meters (e.g., --target 2.0 0.0)'
    )
    parser.add_argument(
        '--explore',
        action='store_true',
        help='Autonomous exploration mode (random targets)'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=120.0,
        help='Navigation timeout in seconds (default: 120)'
    )
    parser.add_argument(
        '--num-targets',
        type=int,
        default=5,
        help='Number of targets for exploration mode (default: 5)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.target and not args.explore:
        parser.print_help()
        print("\nExample usage:")
        print("  python scripts/autonomous_nav.py --target 2.0 0.0")
        print("  python scripts/autonomous_nav.py --explore")
        return

    # Create navigator
    try:
        nav = AutonomousNavigator()
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    try:
        if args.explore:
            # Exploration mode
            nav.explore(num_targets=args.num_targets)
        else:
            # Single target mode
            nav.set_target(args.target[0], args.target[1])
            nav.navigate_to_target(timeout=args.timeout)
    finally:
        nav.cleanup()


if __name__ == "__main__":
    main()
