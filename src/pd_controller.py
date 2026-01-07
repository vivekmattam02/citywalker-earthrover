"""
PD Controller

Converts waypoint predictions to velocity commands for EarthRover.

Usage:
    from pd_controller import PDController

    controller = PDController()
    linear, angular = controller.compute(waypoint_x, waypoint_y)
"""

import math
import numpy as np


class PDController:
    """
    Proportional-Derivative controller for waypoint following.

    Takes a target waypoint in robot frame and outputs velocity commands.
    """

    def __init__(
        self,
        kp_linear=2.0,
        kd_linear=0.1,
        kp_angular=2.0,
        kd_angular=0.1,
        max_linear=1.0,
        max_angular=1.0,
        min_linear=-1.0,
        min_angular=-1.0
    ):
        """
        Initialize PD controller.

        Args:
            kp_linear: Proportional gain for linear velocity
            kd_linear: Derivative gain for linear velocity
            kp_angular: Proportional gain for angular velocity
            kd_angular: Derivative gain for angular velocity
            max_linear: Maximum linear velocity (default 1.0 for EarthRover)
            max_angular: Maximum angular velocity (default 1.0 for EarthRover)
            min_linear: Minimum linear velocity
            min_angular: Minimum angular velocity
        """
        self.kp_linear = kp_linear
        self.kd_linear = kd_linear
        self.kp_angular = kp_angular
        self.kd_angular = kd_angular

        self.max_linear = max_linear
        self.max_angular = max_angular
        self.min_linear = min_linear
        self.min_angular = min_angular

        # Previous errors for derivative term
        self.prev_distance_error = 0.0
        self.prev_angle_error = 0.0

        # Previous time for dt calculation
        self.prev_time = None

    def compute(self, waypoint_x, waypoint_y, dt=0.1):
        """
        Compute velocity commands to reach waypoint.

        Args:
            waypoint_x: X position of waypoint in robot frame (forward is positive)
            waypoint_y: Y position of waypoint in robot frame (left is positive)
            dt: Time step in seconds (default 0.1 = 10Hz)

        Returns:
            (linear_vel, angular_vel): Velocities in range [-1, 1]
        """
        # Calculate distance and angle to waypoint
        distance = math.sqrt(waypoint_x**2 + waypoint_y**2)
        angle = math.atan2(waypoint_y, waypoint_x)

        # Distance error (we want to reduce distance to 0)
        distance_error = distance

        # Angle error (we want to face the waypoint)
        angle_error = angle

        # Derivative terms
        d_distance = (distance_error - self.prev_distance_error) / dt
        d_angle = (angle_error - self.prev_angle_error) / dt

        # PD control
        linear_vel = self.kp_linear * distance_error + self.kd_linear * d_distance
        angular_vel = self.kp_angular * angle_error + self.kd_angular * d_angle

        # If waypoint is behind us (x < 0), we might want to back up or turn around
        # For now, reduce linear speed if we need to turn a lot
        if abs(angle_error) > math.pi / 2:
            # Waypoint is behind us, slow down linear, focus on turning
            linear_vel *= 0.2

        # Clip to valid range
        linear_vel = np.clip(linear_vel, self.min_linear, self.max_linear)
        angular_vel = np.clip(angular_vel, self.min_angular, self.max_angular)

        # Save for next iteration
        self.prev_distance_error = distance_error
        self.prev_angle_error = angle_error

        return float(linear_vel), float(angular_vel)

    def reset(self):
        """Reset controller state."""
        self.prev_distance_error = 0.0
        self.prev_angle_error = 0.0
        self.prev_time = None

    def compute_from_waypoints(self, waypoints, waypoint_index=0, dt=0.1):
        """
        Compute velocity from array of waypoints.

        Args:
            waypoints: numpy array of shape (N, 2) with (x, y) positions
            waypoint_index: Which waypoint to target (default 0 = first/closest)
            dt: Time step in seconds

        Returns:
            (linear_vel, angular_vel): Velocities in range [-1, 1]
        """
        if waypoints is None or len(waypoints) == 0:
            return 0.0, 0.0

        # Get target waypoint
        wp = waypoints[min(waypoint_index, len(waypoints) - 1)]
        return self.compute(wp[0], wp[1], dt)


class StopController:
    """
    Simple controller that always outputs stop command.
    Used when we've arrived at destination.
    """

    def compute(self, *args, **kwargs):
        return 0.0, 0.0

    def reset(self):
        pass


# Quick test if run directly
if __name__ == "__main__":
    print("Testing PDController...")

    controller = PDController()

    # Test 1: Waypoint directly ahead
    print("\n[Test 1] Waypoint 1m ahead:")
    linear, angular = controller.compute(1.0, 0.0)
    print(f"  linear={linear:.3f}, angular={angular:.3f}")
    print(f"  Expected: linear > 0, angular ~ 0")

    controller.reset()

    # Test 2: Waypoint to the left
    print("\n[Test 2] Waypoint 1m to the left:")
    linear, angular = controller.compute(0.0, 1.0)
    print(f"  linear={linear:.3f}, angular={angular:.3f}")
    print(f"  Expected: linear > 0, angular > 0 (turn left)")

    controller.reset()

    # Test 3: Waypoint to the right
    print("\n[Test 3] Waypoint 1m to the right:")
    linear, angular = controller.compute(0.0, -1.0)
    print(f"  linear={linear:.3f}, angular={angular:.3f}")
    print(f"  Expected: linear > 0, angular < 0 (turn right)")

    controller.reset()

    # Test 4: Waypoint behind
    print("\n[Test 4] Waypoint 1m behind:")
    linear, angular = controller.compute(-1.0, 0.0)
    print(f"  linear={linear:.3f}, angular={angular:.3f}")
    print(f"  Expected: linear small, angular ~ pi or -pi (turn around)")

    controller.reset()

    # Test 5: With waypoints array
    print("\n[Test 5] Using waypoints array:")
    waypoints = np.array([
        [0.1, 0.02],
        [0.2, 0.04],
        [0.3, 0.05],
        [0.4, 0.06],
        [0.5, 0.07]
    ])
    linear, angular = controller.compute_from_waypoints(waypoints, waypoint_index=0)
    print(f"  Targeting first waypoint: linear={linear:.3f}, angular={angular:.3f}")

    print("\nPDController test passed!")
