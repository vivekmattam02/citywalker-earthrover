"""
Bug2 Algorithm - Classic Wall-Following Navigation

This implements the Bug2 algorithm, a proven navigation algorithm from robotics literature.
It's been used since the 1980s and is well-tested.

Algorithm:
1. Move toward goal
2. If obstacle hit, follow wall (left-hand rule)
3. When back on line to goal, continue toward goal
4. Repeat until goal reached

Reference: Lumelsky & Stepanov, "Path-planning strategies for a point mobile automaton"
"""

import sys
import os
import time
import numpy as np
import cv2
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from earthrover_interface import EarthRoverInterface
from visual_odometry import VisualOdometry
from pd_controller import PDController


class BugState(Enum):
    """Bug algorithm states."""
    MOVE_TO_GOAL = 1
    FOLLOW_WALL = 2
    CIRCUMNAVIGATE = 3


class Bug2Navigator:
    """
    Bug2 Algorithm - Classic wall-following navigation.
    
    This is a proven algorithm from robotics literature that's been used
    successfully for decades. It's simple, robust, and actually works.
    """
    
    def __init__(self):
        print("=" * 60)
        print("BUG2 ALGORITHM - Classic Wall-Following Navigation")
        print("=" * 60)
        print("Algorithm: Proven since 1980s, used in real robots")
        print()
        
        print("[1] Connecting to robot...")
        self.rover = EarthRoverInterface(timeout=10.0)
        if not self.rover.connect():
            print("FAILED - Check SDK server and browser")
            sys.exit(1)
        
        print("[2] Initializing Visual Odometry...")
        self.vo = VisualOdometry(scale=0.03, n_features=500)
        
        print("[3] Initializing controller...")
        self.controller = PDController(
            kp_linear=1.2,
            kp_angular=2.0,
            kd_angular=0.1,
            max_linear=0.4,
            max_angular=0.5
        )
        
        # Bug2 state
        self.state = BugState.MOVE_TO_GOAL
        self.start_x = 0.0  # Where we started toward current goal (for m-line)
        self.start_y = 0.0
        self.goal_x = 5.0  # Default goal: 5m forward
        self.goal_y = 0.0
        self.hit_point = None  # Where we hit the obstacle
        self.leave_point = None  # Where we left the obstacle
        self.following_wall = False
        self.wall_follow_steps = 0  # Steps spent in wall-follow mode
        self.min_wall_follow_steps = 30  # Minimum before checking m-line exit
        self.wall_follow_direction = 1  # 1 = left, -1 = right

        # Obstacle detection
        self.obstacle_threshold = 0.4  # meters
        self.side_sensor_range = 0.3  # meters
        
        print()
        print("Bug2 Navigator ready!")
        print("Strategy: Move to goal, follow wall if blocked")
        print("Press Ctrl+C to stop")
        print("-" * 60)
    
    def set_goal(self, x: float, y: float):
        """Set navigation goal."""
        self.start_x = self.vo.x  # Remember where we are when goal is set (for m-line)
        self.start_y = self.vo.y
        self.goal_x = x
        self.goal_y = y
        self.state = BugState.MOVE_TO_GOAL
        self.hit_point = None
        self.leave_point = None
        self.wall_follow_steps = 0
        print(f"Goal set: ({x:.2f}, {y:.2f}) from ({self.start_x:.2f}, {self.start_y:.2f})")
    
    def detect_obstacle(self, frame, x: float, y: float, heading: float) -> dict:
        """
        Obstacle detection using edge density in center/left/right regions.

        Returns dict with:
            'ahead': True if obstacle in center
            'left':  edge density on left side (higher = more obstacle-like)
            'right': edge density on right side
        """
        result = {'ahead': False, 'left': 0.0, 'right': 0.0}

        if frame is None:
            return result

        h, w = frame.shape[:2]

        # Lower portion of frame (where obstacles appear at ground level)
        roi = frame[int(h * 0.55):, :]

        if roi.size == 0:
            return result

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        rh, rw = edges.shape[:2]

        # Split into left / center / right thirds
        third = rw // 3
        left_edges = edges[:, :third]
        center_edges = edges[:, third:2 * third]
        right_edges = edges[:, 2 * third:]

        left_density = np.sum(left_edges > 0) / max(left_edges.size, 1)
        center_density = np.sum(center_edges > 0) / max(center_edges.size, 1)
        right_density = np.sum(right_edges > 0) / max(right_edges.size, 1)

        result['left'] = left_density
        result['right'] = right_density

        # Center obstacle: high edge density OR very dark
        center_gray = gray[:, third:2 * third]
        mean_brightness = np.mean(center_gray)

        if center_density > 0.15 or mean_brightness < 50:
            result['ahead'] = True

        return result
    
    def get_distance_to_goal(self, x: float, y: float) -> float:
        """Distance to goal."""
        return np.sqrt((self.goal_x - x)**2 + (self.goal_y - y)**2)
    
    def is_on_mline(self, x: float, y: float) -> bool:
        """
        Check if robot is back on the m-line (line from START to goal).
        This is key to Bug2 algorithm.

        Bug2 m-line = straight line from the position where we set the goal
        to the goal itself. We also require the robot to be closer to the
        goal than the hit point, to avoid exiting wall-follow prematurely.
        """
        # M-line: line from START to goal (NOT from hit point)
        dx_goal = self.goal_x - self.start_x
        dy_goal = self.goal_y - self.start_y

        # Vector from start to current position
        dx_curr = x - self.start_x
        dy_curr = y - self.start_y

        # Cross product gives distance from line
        cross = dx_goal * dy_curr - dy_goal * dx_curr

        line_len = np.sqrt(dx_goal**2 + dy_goal**2)
        if line_len < 0.01:
            return False

        dist_to_line = abs(cross) / line_len

        # Must be closer to goal than when we hit the wall
        if self.hit_point:
            hit_dist = self.get_distance_to_goal(self.hit_point[0], self.hit_point[1])
            curr_dist = self.get_distance_to_goal(x, y)
            if curr_dist >= hit_dist:
                return False

        return dist_to_line < 0.2  # Within 20cm of line
    
    def navigate_step(self, frame, x: float, y: float, heading: float):
        """
        One step of Bug2 algorithm.
        Returns (linear, angular) velocities.
        """
        dist_to_goal = self.get_distance_to_goal(x, y)

        # Check if goal reached
        if dist_to_goal < 0.3:
            return (0.0, 0.0)  # Stop

        # Detect obstacle (returns dict with 'ahead', 'left', 'right' densities)
        obs = self.detect_obstacle(frame, x, y, heading)
        obstacle_ahead = obs['ahead']

        if self.state == BugState.MOVE_TO_GOAL:
            if obstacle_ahead:
                # Hit obstacle - switch to wall following
                self.state = BugState.FOLLOW_WALL
                self.hit_point = (x, y)
                self.following_wall = True
                self.wall_follow_steps = 0
                print(f"\n[!] Obstacle detected at ({x:.2f}, {y:.2f})! Following wall")
                # Turn left to start following
                return (0.0, 0.4)

            else:
                # Move toward goal
                goal_rel_x = self.goal_x - x
                goal_rel_y = self.goal_y - y

                # Rotate to robot frame
                cos_h = np.cos(heading)
                sin_h = np.sin(heading)
                robot_x = goal_rel_x * cos_h + goal_rel_y * sin_h
                robot_y = -goal_rel_x * sin_h + goal_rel_y * cos_h

                linear, angular = self.controller.compute(robot_x, robot_y)
                return (linear, angular)

        elif self.state == BugState.FOLLOW_WALL:
            self.wall_follow_steps += 1

            # Check m-line exit (only after minimum steps to prevent oscillation)
            if not obstacle_ahead:
                if (self.wall_follow_steps > self.min_wall_follow_steps
                        and self.hit_point
                        and self.is_on_mline(x, y)):
                    self.state = BugState.MOVE_TO_GOAL
                    self.leave_point = (x, y)
                    self.following_wall = False
                    print(f"\n[!] Back on m-line at ({x:.2f}, {y:.2f})! Resuming toward goal")
                    goal_rel_x = self.goal_x - x
                    goal_rel_y = self.goal_y - y
                    cos_h = np.cos(heading)
                    sin_h = np.sin(heading)
                    robot_x = goal_rel_x * cos_h + goal_rel_y * sin_h
                    robot_y = -goal_rel_x * sin_h + goal_rel_y * cos_h
                    linear, angular = self.controller.compute(robot_x, robot_y)
                    return (linear, angular)

            # Wall following with left/right edge-density feedback
            # Left-hand rule: keep wall on left side
            left_density = obs['left']
            right_density = obs['right']

            if obstacle_ahead:
                # Wall directly ahead — turn right
                return (0.05, -0.35)
            elif left_density > 0.12:
                # Wall on left (good) — go forward, steer slightly right
                # to maintain distance instead of crashing into it
                return (0.25, -0.05)
            elif left_density < 0.04:
                # Lost the wall on left — turn left to re-find it
                return (0.15, 0.25)
            else:
                # Wall at moderate distance on left — go straight
                return (0.25, 0.0)

        return (0.2, 0.0)  # Default: move forward
    
    def explore(self, max_time: float = 600.0):
        """Main exploration loop using Bug2 algorithm."""
        start_time = time.time()
        step = 0
        x, y = 0.0, 0.0  # Initialize so finally block is safe

        # Set exploration goals in sequence
        goals = [
            (3.0, 0.0),   # 3m forward
            (3.0, 2.0),   # Then 2m left
            (0.0, 2.0),   # Then back
            (0.0, 0.0),   # Return to start
        ]
        goal_index = 0
        self.set_goal(*goals[0])

        try:
            while time.time() - start_time < max_time:
                step += 1

                # Get camera frame
                frame = self.rover.get_camera_frame()
                if frame is None:
                    self.rover.send_control(0.0, 0.0)
                    time.sleep(0.1)
                    continue

                # Update visual odometry
                success, x, y, matches = self.vo.process_frame(frame)
                if not success:
                    self.rover.send_control(0.0, 0.0)
                    time.sleep(0.1)
                    continue

                heading = self.vo.theta

                # Check if current goal reached
                dist_to_goal = self.get_distance_to_goal(x, y)
                if dist_to_goal < 0.3:
                    # Goal reached - move to next goal
                    goal_index += 1
                    if goal_index < len(goals):
                        self.set_goal(*goals[goal_index])
                        time.sleep(0.5)
                        continue
                    else:
                        print("\n[!] All goals reached!")
                        break

                # Bug2 navigation step
                linear, angular = self.navigate_step(frame, x, y, heading)

                # Send control
                self.rover.send_control(linear, angular)

                # Status (every 10 steps = ~1 second at 10Hz)
                if step % 10 == 0:
                    state_str = "MOVE_TO_GOAL" if self.state == BugState.MOVE_TO_GOAL else f"FOLLOW_WALL({self.wall_follow_steps})"
                    print(f"\r[Step {step}] Pos: ({x:.2f}, {y:.2f})  "
                          f"Goal: ({self.goal_x:.2f}, {self.goal_y:.2f})  "
                          f"Dist: {dist_to_goal:.2f}m  "
                          f"State: {state_str}  "
                          f"Vel: L={linear:.2f}, A={angular:.2f}  ",
                          end="", flush=True)

                time.sleep(0.1)  # 10 Hz — realistic for ORB + Essential Matrix processing

        except KeyboardInterrupt:
            print("\n[!] Exploration stopped by user")

        finally:
            self.rover.stop()
            print(f"\n[*] Final position: ({x:.2f}, {y:.2f})")
            print(f"[*] Final goal distance: {self.get_distance_to_goal(x, y):.2f}m")


def main():
    navigator = Bug2Navigator()
    navigator.explore(max_time=600.0)


if __name__ == "__main__":
    main()
