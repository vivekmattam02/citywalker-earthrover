"""
Autonomous Frontier Exploration - Classic Robotics Approach

No AI models, just good old-fashioned robotics:
- Visual Odometry for localization
- Occupancy grid mapping
- Frontier-based exploration
- Simple obstacle detection

USAGE:
    python scripts/autonomous_exploration.py
    
    Press Ctrl+C to stop and save the map
"""

import sys
import os
import time
import numpy as np
import cv2
from collections import deque
from typing import Tuple, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from earthrover_interface import EarthRoverInterface
from visual_odometry import VisualOdometry
from pd_controller import PDController


class OccupancyGrid:
    """2D occupancy grid for mapping."""
    
    def __init__(self, resolution: float = 0.1, size: int = 200):
        """
        Args:
            resolution: Meters per cell (0.1 = 10cm cells)
            size: Grid size in cells (200 = 20m x 20m map)
        """
        self.resolution = resolution
        self.size = size
        self.origin = size // 2  # Robot starts at center
        
        # Grid values: -1=unknown, 0=free, 100=occupied
        self.grid = np.full((size, size), -1, dtype=np.int8)
        
        # Mark origin as free
        self.grid[self.origin, self.origin] = 0
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid indices."""
        grid_x = int(x / self.resolution) + self.origin
        grid_y = int(y / self.resolution) + self.origin
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (meters)."""
        x = (grid_x - self.origin) * self.resolution
        y = (grid_y - self.origin) * self.resolution
        return x, y
    
    def is_valid(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid coordinates are within bounds."""
        return 0 <= grid_x < self.size and 0 <= grid_y < self.size
    
    def mark_free(self, x: float, y: float, radius: float = 0.3):
        """Mark area around position as free space."""
        grid_x, grid_y = self.world_to_grid(x, y)
        cells = int(radius / self.resolution)
        
        for dx in range(-cells, cells + 1):
            for dy in range(-cells, cells + 1):
                gx, gy = grid_x + dx, grid_y + dy
                if self.is_valid(gx, gy):
                    # Only mark as free if not already occupied
                    if self.grid[gx, gy] != 100:
                        self.grid[gx, gy] = 0
    
    def mark_obstacle(self, x: float, y: float, radius: float = 0.2):
        """Mark area as occupied."""
        grid_x, grid_y = self.world_to_grid(x, y)
        cells = int(radius / self.resolution)
        
        for dx in range(-cells, cells + 1):
            for dy in range(-cells, cells + 1):
                gx, gy = grid_x + dx, grid_y + dy
                if self.is_valid(gx, gy):
                    self.grid[gx, gy] = 100
    
    def ray_trace(self, x0: float, y0: float, x1: float, y1: float):
        """
        Trace ray from (x0,y0) to (x1,y1), marking free space along the way
        and obstacle at the end.
        """
        gx0, gy0 = self.world_to_grid(x0, y0)
        gx1, gy1 = self.world_to_grid(x1, y1)
        
        # Bresenham's line algorithm
        points = self._bresenham_line(gx0, gy0, gx1, gy1)
        
        # Mark all points except last as free
        for gx, gy in points[:-1]:
            if self.is_valid(gx, gy) and self.grid[gx, gy] != 100:
                self.grid[gx, gy] = 0
        
        # Mark last point as obstacle
        if len(points) > 0:
            gx, gy = points[-1]
            if self.is_valid(gx, gy):
                self.grid[gx, gy] = 100
    
    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    def find_frontiers(self, robot_x: float, robot_y: float, max_distance: float = 5.0) -> List[Tuple[float, float]]:
        """
        Find frontier cells (boundaries between free and unknown space).
        Returns list of frontier positions in world coordinates.
        """
        frontiers = []
        robot_gx, robot_gy = self.world_to_grid(robot_x, robot_y)
        
        for gx in range(self.size):
            for gy in range(self.size):
                # Check if this is a frontier cell
                if self._is_frontier(gx, gy):
                    wx, wy = self.grid_to_world(gx, gy)
                    
                    # Check distance from robot
                    dist = np.sqrt((wx - robot_x)**2 + (wy - robot_y)**2)
                    if dist < max_distance:
                        frontiers.append((wx, wy))
        
        return frontiers
    
    def _is_frontier(self, gx: int, gy: int) -> bool:
        """Check if a cell is a frontier (free cell adjacent to unknown)."""
        if not self.is_valid(gx, gy):
            return False
        
        # Must be free space
        if self.grid[gx, gy] != 0:
            return False
        
        # Check if adjacent to unknown space
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = gx + dx, gy + dy
                if self.is_valid(nx, ny) and self.grid[nx, ny] == -1:
                    return True
        
        return False
    
    def visualize(self) -> np.ndarray:
        """Create visualization of the map."""
        vis = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # Unknown = gray, Free = white, Occupied = black
        vis[self.grid == -1] = [128, 128, 128]  # Gray
        vis[self.grid == 0] = [255, 255, 255]   # White
        vis[self.grid == 100] = [0, 0, 0]       # Black
        
        return vis


class ObstacleDetector:
    """Simple vision-based obstacle detection."""
    
    def __init__(self, detection_range: float = 2.0):
        self.detection_range = detection_range
    
    def detect(self, frame: np.ndarray, robot_x: float, robot_y: float, robot_heading: float) -> List[Tuple[float, float]]:
        """
        Detect obstacles in camera frame.
        Returns list of obstacle positions in world coordinates.
        
        Simple approach:
        - Detect edges/dark regions in lower part of image
        - Map to approximate world positions
        """
        if frame is None:
            return []
        
        h, w = frame.shape[:2]
        
        # Focus on lower half (ground level)
        roi = frame[h//2:, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect dark regions (potential obstacles)
        _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Combine
        obstacles_mask = cv2.bitwise_or(edges, dark)
        
        # Simple heuristic: divide image into segments
        obstacles = []
        n_segments = 5
        segment_width = w // n_segments
        
        for i in range(n_segments):
            segment = obstacles_mask[:, i*segment_width:(i+1)*segment_width]
            obstacle_density = np.sum(segment > 0) / segment.size
            
            # If enough obstacle pixels, mark as obstacle
            if obstacle_density > 0.15:  # Threshold
                # Map to world coordinates
                # Center segment: straight ahead
                # Left segments: left side
                # Right segments: right side
                angle_offset = (i - n_segments/2) * 0.3  # ~17 degrees per segment
                angle = robot_heading + angle_offset
                
                # Place obstacle at detection range
                obs_x = robot_x + self.detection_range * np.cos(angle)
                obs_y = robot_y + self.detection_range * np.sin(angle)
                
                obstacles.append((obs_x, obs_y))
        
        return obstacles


class FrontierExplorer:
    """Autonomous frontier-based exploration."""
    
    def __init__(self):
        print("=" * 60)
        print("AUTONOMOUS FRONTIER EXPLORATION")
        print("=" * 60)
        print()
        
        # Initialize components
        print("[1] Connecting to robot...")
        self.rover = EarthRoverInterface(timeout=10.0)
        if not self.rover.connect():
            print("FAILED - Check SDK server and browser")
            sys.exit(1)
        
        print("[2] Initializing Visual Odometry...")
        self.vo = VisualOdometry(scale=0.05)
        
        print("[3] Initializing occupancy grid...")
        self.map = OccupancyGrid(resolution=0.1, size=200)  # 20m x 20m map
        
        print("[4] Initializing obstacle detector...")
        self.obstacle_detector = ObstacleDetector(detection_range=1.5)
        
        print("[5] Initializing controller...")
        self.controller = PDController(
            kp_linear=0.5,
            kp_angular=1.2,
            kd_angular=0.05,
            max_linear=0.3,
            max_angular=0.4
        )
        
        # State
        self.current_target = None
        self.stuck_counter = 0
        self.last_position = (0.0, 0.0)
        
        print()
        print("Explorer ready!")
        print("Press Ctrl+C to stop and save map")
        print("-" * 60)
    
    def explore(self, max_time: float = 300.0):
        """Main exploration loop."""
        start_time = time.time()
        step = 0
        
        try:
            while time.time() - start_time < max_time:
                step += 1
                
                # Get camera frame
                frame = self.rover.get_camera_frame()
                if frame is None:
                    print("No camera frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Update visual odometry
                self.vo.update(frame)
                x, y, theta = self.vo.x, self.vo.y, self.vo.theta
                
                # Mark current position as free
                self.map.mark_free(x, y)
                
                # Detect obstacles
                obstacles = self.obstacle_detector.detect(frame, x, y, theta)
                for obs_x, obs_y in obstacles:
                    self.map.mark_obstacle(obs_x, obs_y)
                    # Also ray trace from robot to obstacle
                    self.map.ray_trace(x, y, obs_x, obs_y)
                
                # Check if we need a new target
                if self.current_target is None or self._reached_target(x, y):
                    frontiers = self.map.find_frontiers(x, y, max_distance=5.0)
                    
                    if len(frontiers) == 0:
                        print("\n[!] No more frontiers - exploration complete!")
                        break
                    
                    # Choose nearest frontier
                    self.current_target = self._choose_frontier(x, y, frontiers)
                    print(f"\n[Step {step}] New target: ({self.current_target[0]:.2f}, {self.current_target[1]:.2f})")
                
                # Navigate to target
                target_x, target_y = self.current_target
                
                # Compute target in robot frame
                rel_x = target_x - x
                rel_y = target_y - y
                
                # Rotate to robot frame
                cos_h = np.cos(theta)
                sin_h = np.sin(theta)
                robot_x = rel_x * cos_h + rel_y * sin_h
                robot_y = -rel_x * sin_h + rel_y * cos_h
                
                # Compute velocities
                linear, angular = self.controller.compute(robot_x, robot_y)
                
                # Safety limits
                safe_linear = np.clip(linear, 0.0, 0.3)
                safe_angular = np.clip(angular, -0.4, 0.4)
                
                # Check for obstacles directly ahead
                if len(obstacles) > 0:
                    # If obstacles detected, reduce speed and adjust
                    safe_linear *= 0.5
                
                # Send control
                self.rover.send_control(safe_linear, safe_angular)
                
                # Check if stuck
                if self._is_stuck(x, y):
                    print(f"\r[Step {step}] STUCK - Finding new frontier...", end="", flush=True)
                    self.current_target = None
                    self.stuck_counter = 0
                    # Turn in place
                    self.rover.send_control(0.0, 0.3)
                    time.sleep(0.5)
                    continue
                
                # Status
                dist_to_target = np.sqrt((target_x - x)**2 + (target_y - y)**2)
                n_frontiers = len(self.map.find_frontiers(x, y, max_distance=5.0))
                
                print(f"\r[Step {step}] Pos: ({x:.2f}, {y:.2f})  "
                      f"Target: ({target_x:.2f}, {target_y:.2f})  "
                      f"Dist: {dist_to_target:.2f}m  "
                      f"Frontiers: {n_frontiers}  "
                      f"Vel: L={safe_linear:.2f}, A={safe_angular:.2f}  ",
                      end="", flush=True)
                
                time.sleep(0.05)  # 20 Hz
            
            # Timeout
            print(f"\n[!] Exploration timeout ({max_time}s)")
        
        except KeyboardInterrupt:
            print("\n[!] Exploration stopped by user")
        
        finally:
            self.rover.stop()
            self._save_map()
    
    def _reached_target(self, x: float, y: float, threshold: float = 0.5) -> bool:
        """Check if robot reached current target."""
        if self.current_target is None:
            return True
        target_x, target_y = self.current_target
        dist = np.sqrt((target_x - x)**2 + (target_y - y)**2)
        return dist < threshold
    
    def _choose_frontier(self, x: float, y: float, frontiers: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Choose best frontier to explore (nearest)."""
        if len(frontiers) == 0:
            return (x, y)
        
        # Sort by distance
        distances = [np.sqrt((fx - x)**2 + (fy - y)**2) for fx, fy in frontiers]
        nearest_idx = np.argmin(distances)
        
        return frontiers[nearest_idx]
    
    def _is_stuck(self, x: float, y: float, threshold: float = 0.05) -> bool:
        """Check if robot is stuck (not moving)."""
        dist_moved = np.sqrt((x - self.last_position[0])**2 + (y - self.last_position[1])**2)
        
        if dist_moved < threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        self.last_position = (x, y)
        
        return self.stuck_counter > 20  # Stuck for 1 second
    
    def _save_map(self):
        """Save the map visualization."""
        print("\n[*] Saving map...")
        map_vis = self.map.visualize()
        
        # Add robot position
        robot_gx, robot_gy = self.map.world_to_grid(self.vo.x, self.vo.y)
        cv2.circle(map_vis, (robot_gy, robot_gx), 3, (0, 255, 0), -1)
        
        # Save
        output_path = "exploration_map.png"
        cv2.imwrite(output_path, map_vis)
        print(f"[*] Map saved to: {output_path}")
        
        # Print stats
        total_cells = self.map.size * self.map.size
        explored_cells = np.sum(self.map.grid >= 0)
        coverage = 100.0 * explored_cells / total_cells
        
        print(f"[*] Exploration statistics:")
        print(f"    - Total area: {self.map.size * self.map.resolution:.1f}m x {self.map.size * self.map.resolution:.1f}m")
        print(f"    - Coverage: {coverage:.1f}%")
        print(f"    - Final position: ({self.vo.x:.2f}, {self.vo.y:.2f})")


def main():
    explorer = FrontierExplorer()
    explorer.explore(max_time=600.0)  # 10 minutes max


if __name__ == "__main__":
    main()
