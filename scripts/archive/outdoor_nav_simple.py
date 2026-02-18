"""
SIMPLE GPS Navigation - No tricks, just go to target

Usage:
    python scripts/outdoor_nav_simple.py --target-lat LAT --target-lon LON
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from citywalker_wrapper import CityWalkerWrapper
from earthrover_interface import EarthRoverInterface
from coordinate_utils import CoordinateTransformer, haversine_distance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-lat', type=float, required=True)
    parser.add_argument('--target-lon', type=float, required=True)
    args = parser.parse_args()

    print("Loading model...")
    model = CityWalkerWrapper()

    print("Connecting to robot...")
    rover = EarthRoverInterface(timeout=30.0)
    if not rover.connect():
        print("Failed to connect")
        return

    print("Initializing...")
    transformer = CoordinateTransformer(history_size=5)

    target_lat = args.target_lat
    target_lon = args.target_lon

    print(f"\nTarget: ({target_lat}, {target_lon})")
    print("Starting navigation...\n")

    image_buffer = []
    step = 0

    try:
        while True:
            step += 1

            # Get frame
            frame = rover.get_camera_frame()
            if frame is None:
                rover.stop()
                time.sleep(0.1)
                continue

            # Get GPS
            lat, lon, heading_rad = rover.get_pose()
            if lat is None or lon is None or heading_rad is None:
                rover.stop()
                time.sleep(0.1)
                continue

            # Update transformer
            transformer.update(lat, lon, heading_rad)

            # Build image buffer
            image_buffer.append(frame)
            if len(image_buffer) > 5:
                image_buffer.pop(0)

            if len(image_buffer) < 5 or not transformer.is_ready():
                print(f"\rBuffering... {len(image_buffer)}/5", end="", flush=True)
                time.sleep(0.1)
                continue

            # Check distance
            distance = haversine_distance(lat, lon, target_lat, target_lon)
            if distance < 1.0:
                print(f"\n\nARRIVED! Distance: {distance:.2f}m")
                rover.stop()
                break

            # Get CityWalker input
            coords = transformer.get_model_input(target_lat, target_lon)
            target_robot = coords[5]  # Target in robot frame

            # Run CityWalker
            images = np.stack(image_buffer, axis=0)
            waypoints, arrival_prob = model.predict(images, coords, step_scale=0.3)

            # PURE TARGET FOLLOWING - ignore CityWalker for now
            # Just go toward the target
            wp_x, wp_y = target_robot[0], target_robot[1]

            # Simple proportional control
            angle_to_target = np.arctan2(wp_y, wp_x)

            # Forward speed based on distance ahead
            if wp_x > 0:
                linear = min(0.4, wp_x * 0.2)
            else:
                linear = 0.0

            # Turn toward target
            angular = np.clip(angle_to_target * 1.5, -0.7, 0.7)

            # Send command
            rover.send_control(linear, angular)

            # Status
            if step % 10 == 0:
                heading_deg = np.degrees(heading_rad)
                print(f"\r[{step}] Dist: {distance:.1f}m | H: {heading_deg:.0f}° | "
                      f"Tgt: ({target_robot[0]:.1f}, {target_robot[1]:.1f})m | "
                      f"V: ({linear:.2f}, {angular:.2f})   ", end="", flush=True)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        rover.stop()
        print("Done.")

if __name__ == "__main__":
    main()
