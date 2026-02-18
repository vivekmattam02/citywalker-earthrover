"""
Navigation Monitor

Shows real-time debug info while you manually control robot via SDK browser.

Open browser: http://localhost:8000/sdk
Drive robot manually and watch this monitor to see:
- GPS position changes
- Distance traveled
- What CityWalker would predict
- Heading changes
"""

import sys
import os
import time
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from earthrover_interface import EarthRoverInterface
from citywalker_wrapper import CityWalkerWrapper
from coordinate_utils import CoordinateTransformer, haversine_distance


def main():
    print("="*70)
    print("NAVIGATION MONITOR - Real-Time Debug")
    print("="*70)
    print("\n1. Open browser: http://localhost:8000/sdk")
    print("2. Use SDK controls to drive robot")
    print("3. Watch this monitor for debug info")
    print("\nPress Ctrl+C to stop\n")

    # Connect
    rover = EarthRoverInterface(timeout=30.0)
    if not rover.connect():
        return

    # Load model
    print("\nLoading CityWalker...")
    model = CityWalkerWrapper()
    print("✓ Ready!\n")

    # Initialize
    transformer = CoordinateTransformer(history_size=5)
    image_buffer = deque(maxlen=5)

    # Get starting position
    start_lat, start_lon, _ = rover.get_pose()
    last_lat, last_lon = start_lat, start_lon

    print("="*70)
    print(f"{'Time':>6} | {'GPS Dist':>8} | {'Move':>6} | {'Heading':>7} | {'CityWalker Prediction'}")
    print("-"*70)

    step = 0
    try:
        while True:
            step += 1

            # Get current state
            lat, lon, heading = rover.get_pose()
            if lat is None:
                time.sleep(0.5)
                continue

            # Update transformer
            transformer.update(lat, lon, heading)

            # Get frame
            frame = rover.get_camera_frame()
            if frame is not None:
                image_buffer.append(frame)

            # Calculate distances
            total_dist = haversine_distance(start_lat, start_lon, lat, lon)
            move_since_last = haversine_distance(last_lat, last_lon, lat, lon)

            # CityWalker prediction (every 10 steps)
            cw_info = ""
            if step % 10 == 0 and len(image_buffer) >= 5 and transformer.is_ready():
                # Set target 5m ahead
                target_x = 5.0 * np.sin(heading)
                target_y = 5.0 * np.cos(heading)
                target_lat = lat + target_y / 111139.0
                target_lon = lon + target_x / (111139.0 * np.cos(np.radians(lat)))

                coords = transformer.get_model_input(target_lat, target_lon)
                images = np.stack(list(image_buffer), axis=0)

                try:
                    waypoints, arrival_prob = model.predict(images, coords, step_scale=0.3)
                    wp = waypoints[2]  # Use waypoint[2]
                    wp_scaled = wp * 8.0  # Scale 8x
                    cw_info = f"WP: ({wp_scaled[0]:5.2f}, {wp_scaled[1]:5.2f})m"
                except:
                    cw_info = "CW: Error"

            # Print status
            print(f"{step*0.5:5.1f}s | {total_dist:7.2f}m | {move_since_last:5.2f}m | {np.degrees(heading):6.1f}° | {cw_info}")

            last_lat, last_lon = lat, lon
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n" + "="*70)
        print("Stopped")
        print(f"Total distance traveled: {total_dist:.2f}m")


if __name__ == "__main__":
    main()
