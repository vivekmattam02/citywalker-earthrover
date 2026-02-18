"""
Keyboard Control with Debug Output

Test robot control and see what CityWalker predicts.

Controls:
  W - Forward
  S - Backward
  A - Turn left
  D - Turn right
  SPACE - Stop
  Q - Quit

  C - Test CityWalker prediction (shows what it would navigate to)
"""

import sys
import os
import time
import numpy as np
from collections import deque
import keyboard

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from earthrover_interface import EarthRoverInterface
from citywalker_wrapper import CityWalkerWrapper
from coordinate_utils import CoordinateTransformer, haversine_distance


def main():
    print("="*60)
    print("KEYBOARD CONTROL + DEBUG")
    print("="*60)

    # Connect to robot
    print("\nConnecting to robot...")
    rover = EarthRoverInterface(timeout=30.0)
    if not rover.connect():
        print("✗ Failed to connect")
        return

    # Load CityWalker
    print("\nLoading CityWalker model...")
    model = CityWalkerWrapper()
    print("✓ Model loaded")

    # Initialize coordinate transformer
    transformer = CoordinateTransformer(history_size=5)
    image_buffer = deque(maxlen=5)

    # Control parameters
    linear_speed = 0.3
    angular_speed = 0.5

    # Get initial position
    lat, lon, heading = rover.get_pose()
    print(f"\nInitial position:")
    print(f"  GPS: ({lat:.6f}, {lon:.6f})")
    print(f"  Heading: {np.degrees(heading):.1f}°")

    # Track distance traveled
    start_lat, start_lon = lat, lon

    print("\n" + "="*60)
    print("CONTROLS")
    print("="*60)
    print("  W - Forward")
    print("  S - Backward")
    print("  A - Turn left")
    print("  D - Turn right")
    print("  SPACE - Stop")
    print("  C - Test CityWalker (predict waypoint)")
    print("  Q - Quit")
    print("="*60)
    print("\nPress keys to control robot...")
    print("(Keep this terminal focused!)\n")

    try:
        while True:
            # Get current state
            lat, lon, heading = rover.get_pose()
            distance_from_start = haversine_distance(start_lat, start_lon, lat, lon)

            # Update transformer
            transformer.update(lat, lon, heading)

            # Get camera frame
            frame = rover.get_camera_frame()
            if frame is not None:
                image_buffer.append(frame)

            # Handle keyboard input
            linear = 0.0
            angular = 0.0

            if keyboard.is_pressed('w'):
                linear = linear_speed
                print(f"\r▲ FORWARD | Dist from start: {distance_from_start:.2f}m | H: {np.degrees(heading):.0f}°   ", end="", flush=True)
            elif keyboard.is_pressed('s'):
                linear = -linear_speed
                print(f"\r▼ BACKWARD | Dist from start: {distance_from_start:.2f}m | H: {np.degrees(heading):.0f}°   ", end="", flush=True)
            elif keyboard.is_pressed('a'):
                angular = angular_speed
                print(f"\r◄ LEFT | Dist from start: {distance_from_start:.2f}m | H: {np.degrees(heading):.0f}°   ", end="", flush=True)
            elif keyboard.is_pressed('d'):
                angular = -angular_speed
                print(f"\r► RIGHT | Dist from start: {distance_from_start:.2f}m | H: {np.degrees(heading):.0f}°   ", end="", flush=True)
            elif keyboard.is_pressed('space'):
                linear = 0.0
                angular = 0.0
                print(f"\r■ STOP | Dist from start: {distance_from_start:.2f}m | H: {np.degrees(heading):.0f}°   ", end="", flush=True)
            elif keyboard.is_pressed('c'):
                # Test CityWalker prediction
                if len(image_buffer) >= 5 and transformer.is_ready():
                    # Set target 5m ahead
                    from coordinate_utils import gps_to_local, rotate_to_robot_frame
                    target_x = 5.0 * np.sin(heading)
                    target_y = 5.0 * np.cos(heading)
                    target_lat = lat + target_y / 111139.0
                    target_lon = lon + target_x / (111139.0 * np.cos(np.radians(lat)))

                    coords = transformer.get_model_input(target_lat, target_lon)
                    images = np.stack(list(image_buffer), axis=0)

                    waypoints, arrival_prob = model.predict(images, coords, step_scale=0.3)

                    print(f"\n\nCityWalker Prediction:")
                    print(f"  Target: 5m ahead")
                    print(f"  Waypoints (raw):")
                    for i, wp in enumerate(waypoints):
                        print(f"    [{i}] ({wp[0]:6.2f}, {wp[1]:6.2f}) meters")
                    print(f"  Waypoint[2] scaled 8x: ({waypoints[2][0]*8:.2f}, {waypoints[2][1]*8:.2f}) meters")
                    print(f"  Arrival prob: {arrival_prob*100:.1f}%")
                    print(f"\nContinue driving...\n")
                else:
                    print("\n⚠ Need 5 frames first - drive around a bit\n")

            elif keyboard.is_pressed('q'):
                print("\n\nQuitting...")
                break

            # Send control
            rover.send_control(linear, angular)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped by Ctrl+C")
    finally:
        rover.stop()
        print("\nRobot stopped")
        print(f"Total distance traveled: {distance_from_start:.2f}m")


if __name__ == "__main__":
    main()
