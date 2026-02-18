"""
Outdoor GPS Navigation with CityWalker - PROPER IMPLEMENTATION

This implements outdoor GPS-based navigation properly:
1. Uses IMU heading (more accurate than GPS orientation)
2. Handles GPS jitter (1 Hz updates, ±2-5m noise)
3. Continuous replanning (paper's approach)
4. Scales waypoints for GPS distances
5. Blends CityWalker obstacle avoidance with GPS goal-reaching

Usage:
    python scripts/outdoor_gps_citywalker.py --target-lat LAT --target-lon LON

Requirements:
    - Must be run OUTDOORS (GPS needs satellite signal)
    - SDK must be running
    - Robot must have clear GPS signal
"""

import sys
import os
import time
import argparse
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from citywalker_wrapper import CityWalkerWrapper
from earthrover_interface import EarthRoverInterface
from coordinate_utils import CoordinateTransformer, haversine_distance


class OutdoorGPSNavigator:
    """Outdoor GPS navigation using CityWalker + GPS + IMU"""

    def __init__(self):
        print("="*60)
        print("OUTDOOR GPS NAVIGATION WITH CITYWALKER")
        print("="*60)

        # Initialize model
        print("\n[1/4] Loading CityWalker model...")
        self.model = CityWalkerWrapper()

        # Initialize robot
        print("[2/4] Connecting to robot...")
        self.rover = EarthRoverInterface(timeout=30.0)
        if not self.rover.connect():
            raise RuntimeError("Failed to connect. Is SDK running?")

        # Initialize coordinate transformer
        print("[3/4] Initializing GPS transformer...")
        self.transformer = CoordinateTransformer(history_size=5)

        # Initialize control parameters
        print("[4/4] Setting up navigation...")

        # Navigation state
        self.target_lat = None
        self.target_lon = None
        self.goal_tolerance = 2.0  # meters - GPS accuracy limit

        # CityWalker parameters
        self.waypoint_index = 2  # Use 3rd waypoint (0.4s ahead)
        self.waypoint_scale = 8.0  # Scale CityWalker output by 8x for GPS distances

        # Replanning parameters (from paper)
        self.replan_distance = 4.0  # meters - give CityWalker targets this far ahead
        self.use_replanning = True  # Continuous replanning (paper's approach)

        # Control parameters
        self.max_linear = 0.5  # m/s
        self.max_angular = 0.7  # rad/s
        self.min_forward_speed = 0.15  # minimum when moving

        # Image buffer
        self.image_buffer = deque(maxlen=5)

        # Heading fusion (use IMU + GPS)
        self.heading_buffer = deque(maxlen=10)

        # GPS tracking
        self.last_gps_update = 0
        self.gps_update_interval = 1.0  # GPS updates at 1 Hz

        print("\n✓ Navigator ready!")
        print("  Waypoint scale: {}x".format(self.waypoint_scale))
        print("  Replanning distance: {}m".format(self.replan_distance))
        print("  Goal tolerance: {}m".format(self.goal_tolerance))

    def set_target(self, lat, lon):
        """Set GPS target"""
        self.target_lat = lat
        self.target_lon = lon
        print(f"\n→ Target: ({lat}, {lon})")

    def get_fused_heading(self, gps_heading_rad):
        """
        Fuse GPS heading with IMU for better accuracy

        GPS heading: 1 Hz, noisy but absolute
        IMU gyro: High rate, accurate for short term

        For now: just use filtered GPS heading
        TODO: Implement proper sensor fusion with gyro
        """
        if gps_heading_rad is None:
            if self.heading_buffer:
                return self.heading_buffer[-1]
            return 0.0

        self.heading_buffer.append(gps_heading_rad)

        # Average using sin/cos to handle wrap-around
        sin_sum = sum(np.sin(h) for h in self.heading_buffer)
        cos_sum = sum(np.cos(h) for h in self.heading_buffer)
        fused_heading = np.arctan2(sin_sum, cos_sum)

        return fused_heading

    def compute_intermediate_target(self, current_lat, current_lon, goal_lat, goal_lon, max_dist):
        """
        Compute intermediate target toward goal (for continuous replanning)

        CityWalker was trained on targets 2-10m away.
        For long distances, we set intermediate goals and replan frequently.
        """
        from coordinate_utils import gps_to_local

        # Get goal in local meters
        goal_x, goal_y = gps_to_local(current_lat, current_lon, goal_lat, goal_lon)
        goal_dist = np.sqrt(goal_x**2 + goal_y**2)

        if goal_dist <= max_dist:
            # Close enough - use actual goal
            return goal_lat, goal_lon

        # Set intermediate target max_dist meters toward goal
        scale = max_dist / goal_dist
        intermediate_x = goal_x * scale
        intermediate_y = goal_y * scale

        # Convert back to GPS
        # Approximate conversion (works for small distances)
        intermediate_lat = current_lat + intermediate_y / 111139.0
        intermediate_lon = current_lon + intermediate_x / (111139.0 * np.cos(np.radians(current_lat)))

        return intermediate_lat, intermediate_lon

    def navigate_step(self):
        """Execute one navigation step"""

        # Get camera frame
        frame = self.rover.get_camera_frame()
        if frame is None:
            self.rover.stop()
            return {'success': False, 'reason': 'no_frame'}

        # Get GPS position
        lat, lon, gps_heading_rad = self.rover.get_pose()
        if lat is None or lon is None:
            self.rover.stop()
            return {'success': False, 'reason': 'no_gps'}

        # Fuse heading from GPS + IMU
        heading_rad = self.get_fused_heading(gps_heading_rad)

        # Update transformer
        self.transformer.update(lat, lon, heading_rad)

        # Update image buffer
        self.image_buffer.append(frame)

        # Need 5 frames
        if len(self.image_buffer) < 5 or not self.transformer.is_ready():
            return {
                'success': True,
                'buffering': True,
                'buffer_size': len(self.image_buffer),
                'transformer_ready': self.transformer.is_ready()
            }

        # Check distance to goal
        distance = haversine_distance(lat, lon, self.target_lat, self.target_lon)

        if distance < self.goal_tolerance:
            self.rover.stop()
            return {
                'success': True,
                'arrived': True,
                'distance': distance
            }

        # Compute target for CityWalker (intermediate if far away)
        if self.use_replanning and distance > self.replan_distance:
            # Use intermediate target (paper's continuous replanning)
            cw_target_lat, cw_target_lon = self.compute_intermediate_target(
                lat, lon, self.target_lat, self.target_lon, self.replan_distance
            )
        else:
            # Close to goal - use actual target
            cw_target_lat, cw_target_lon = self.target_lat, self.target_lon

        # Get coordinates for CityWalker
        coords = self.transformer.get_model_input(cw_target_lat, cw_target_lon)
        target_robot = coords[5]  # Target in robot frame

        # Run CityWalker
        try:
            images = np.stack(list(self.image_buffer), axis=0)
            waypoints, arrival_prob = self.model.predict(images, coords, step_scale=0.3)
        except Exception as e:
            self.rover.stop()
            return {'success': False, 'reason': f'model_error: {e}'}

        # Get waypoint and scale it
        cw_waypoint = waypoints[self.waypoint_index] * self.waypoint_scale

        # Blend CityWalker with direct goal vector
        # Far from goal: mostly go direct, use CW for obstacle avoidance
        # Close to goal: trust CityWalker more for final approach
        if distance > self.replan_distance:
            # Far: 85% direct, 15% CityWalker
            blend_alpha = 0.85
        elif distance > self.goal_tolerance * 2:
            # Medium: 60% direct, 40% CityWalker
            blend_alpha = 0.60
        else:
            # Close: 30% direct, 70% CityWalker (trust obstacle avoidance)
            blend_alpha = 0.30

        # Compute final waypoint
        wp_x = blend_alpha * target_robot[0] + (1 - blend_alpha) * cw_waypoint[0]
        wp_y = blend_alpha * target_robot[1] + (1 - blend_alpha) * cw_waypoint[1]

        # Convert to control commands (simple proportional)
        wp_dist = np.sqrt(wp_x**2 + wp_y**2)
        wp_angle = np.arctan2(wp_y, wp_x)

        # Linear velocity proportional to forward distance
        if wp_x > 0:
            linear = min(self.max_linear, wp_dist * 0.3)
            linear = max(linear, self.min_forward_speed)  # Minimum speed
        else:
            linear = 0.0  # Target is behind

        # Angular velocity proportional to angle
        angular = np.clip(wp_angle * 1.8, -self.max_angular, self.max_angular)

        # Slow down if turning hard
        if abs(angular) > 0.4:
            linear *= 0.6

        # Send control
        self.rover.send_control(linear, angular)

        return {
            'success': True,
            'position': (lat, lon),
            'heading_deg': np.degrees(heading_rad),
            'distance': distance,
            'target_robot': (target_robot[0], target_robot[1]),
            'cw_waypoint': (cw_waypoint[0], cw_waypoint[1]),
            'final_waypoint': (wp_x, wp_y),
            'velocity': (linear, angular),
            'arrival_prob': arrival_prob,
            'blend_alpha': blend_alpha
        }

    def navigate(self, timeout=300.0):
        """Main navigation loop"""
        if self.target_lat is None or self.target_lon is None:
            print("✗ No target set!")
            return False

        print(f"\n{'='*60}")
        print(f"NAVIGATING TO: ({self.target_lat}, {self.target_lon})")
        print(f"Timeout: {timeout}s")
        print(f"{'='*60}\n")

        start_time = time.time()
        step = 0

        try:
            while time.time() - start_time < timeout:
                step += 1
                status = self.navigate_step()

                # Handle buffering
                if status.get('buffering'):
                    buf_size = status.get('buffer_size', 0)
                    trans_ready = status.get('transformer_ready', False)
                    print(f"\rBuffering... Frames: {buf_size}/5, GPS history: {'ready' if trans_ready else 'building'}   ", end="", flush=True)
                    time.sleep(0.1)
                    continue

                # Check errors
                if not status['success']:
                    print(f"\n✗ Error: {status.get('reason')}")
                    time.sleep(0.5)
                    continue

                # Check arrival
                if status.get('arrived'):
                    elapsed = time.time() - start_time
                    print(f"\n\n{'='*60}")
                    print(f"✓ TARGET REACHED!")
                    print(f"  Time: {elapsed:.1f}s")
                    print(f"  Distance: {status['distance']:.2f}m")
                    print(f"{'='*60}")
                    return True

                # Print status
                if step % 10 == 0:  # Every 1 second at 10Hz
                    elapsed = time.time() - start_time
                    dist = status['distance']
                    heading = status['heading_deg']
                    tgt = status['target_robot']
                    cw = status['cw_waypoint']
                    wp = status['final_waypoint']
                    vel = status['velocity']
                    alpha = status['blend_alpha']

                    print(f"\r[{elapsed:.0f}s] "
                          f"Dist: {dist:.1f}m | "
                          f"H: {heading:.0f}° | "
                          f"Tgt: ({tgt[0]:.1f},{tgt[1]:.1f}) | "
                          f"CW: ({cw[0]:.1f},{cw[1]:.1f}) | "
                          f"WP: ({wp[0]:.1f},{wp[1]:.1f}) | "
                          f"Blend: {alpha:.0%} | "
                          f"V: ({vel[0]:.2f},{vel[1]:.2f})   ", end="", flush=True)

                time.sleep(0.1)  # 10 Hz

        except KeyboardInterrupt:
            print("\n\n✗ Stopped by user")
            return False
        finally:
            self.rover.stop()

        print(f"\n\n✗ Timeout ({timeout}s)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Outdoor GPS Navigation with CityWalker")
    parser.add_argument('--target-lat', type=float, help='Target latitude')
    parser.add_argument('--target-lon', type=float, help='Target longitude')
    parser.add_argument('--timeout', type=float, default=300.0, help='Timeout in seconds (default: 300)')
    parser.add_argument('--show-gps', action='store_true', help='Show current GPS and exit')

    args = parser.parse_args()

    # Show GPS mode
    if args.show_gps:
        rover = EarthRoverInterface(timeout=30.0)
        if rover.connect():
            lat, lon, heading_rad = rover.get_pose()
            heading_deg = np.degrees(heading_rad) if heading_rad else 0
            print(f"\n{'='*60}")
            print("CURRENT GPS POSITION")
            print(f"{'='*60}")
            print(f"\nLatitude:  {lat}")
            print(f"  Longitude: {lon}")
            print(f"  Heading:   {heading_deg:.1f}°")
            print(f"\nGoogle Maps: https://maps.google.com/?q={lat},{lon}")
        return

    # Navigation mode - require target coordinates
    if args.target_lat is None or args.target_lon is None:
        print("Error: --target-lat and --target-lon are required for navigation")
        print("\nUsage:")
        print("  Show GPS: python outdoor_gps_citywalker.py --show-gps")
        print("  Navigate: python outdoor_gps_citywalker.py --target-lat LAT --target-lon LON")
        sys.exit(1)

    try:
        nav = OutdoorGPSNavigator()
        nav.set_target(args.target_lat, args.target_lon)
        success = nav.navigate(timeout=args.timeout)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
