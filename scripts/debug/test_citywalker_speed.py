"""
Test CityWalker Inference Speed

Tests how fast CityWalker generates waypoints to understand timing constraints.
This answers: Can the robot control loop keep up with CityWalker?
"""

import sys
import os
import time
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from citywalker_wrapper import CityWalkerWrapper
from earthrover_interface import EarthRoverInterface
from coordinate_utils import CoordinateTransformer


def main():
    print("="*70)
    print("CITYWALKER INFERENCE SPEED TEST")
    print("="*70)
    print("\nThis tests how fast CityWalker generates waypoints.")
    print("Critical for syncing robot control with model predictions.\n")

    # Load model
    print("[1/3] Loading CityWalker...")
    start = time.time()
    model = CityWalkerWrapper()
    load_time = time.time() - start
    print(f"✓ Model loaded in {load_time:.2f}s\n")

    # Connect to robot for camera
    print("[2/3] Connecting to robot...")
    rover = EarthRoverInterface(timeout=30.0)
    if not rover.connect():
        print("✗ Failed to connect")
        return
    print("✓ Connected\n")

    # Get initial position
    lat, lon, heading = rover.get_pose()
    if lat is None:
        print("✗ No GPS data")
        return

    print(f"Starting position: ({lat:.6f}, {lon:.6f})")
    print(f"Heading: {np.degrees(heading):.1f}°\n")

    # Initialize transformer
    transformer = CoordinateTransformer(history_size=5)
    transformer.update(lat, lon, heading)

    # Collect 5 frames
    print("[3/3] Collecting 5 camera frames...")
    image_buffer = deque(maxlen=5)

    for i in range(5):
        frame = rover.get_camera_frame()
        if frame is None:
            print(f"  Frame {i+1}/5: ✗ Failed")
            return
        image_buffer.append(frame)
        print(f"  Frame {i+1}/5: ✓ {frame.shape}")
        time.sleep(0.1)

    print("\n" + "="*70)
    print("INFERENCE SPEED TEST (20 iterations)")
    print("="*70)

    # Set target 10m ahead
    target_x = 10.0 * np.sin(heading)
    target_y = 10.0 * np.cos(heading)
    target_lat = lat + target_y / 111139.0
    target_lon = lon + target_x / (111139.0 * np.cos(np.radians(lat)))

    coords = transformer.get_model_input(target_lat, target_lon)
    images = np.stack(list(image_buffer), axis=0)

    print(f"\nInput shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Coords: {coords.shape}")
    print(f"\nRunning 20 inference iterations...\n")

    # Warm-up (first inference is slower)
    print("Warm-up inference...")
    _ = model.predict(images, coords, step_scale=0.3)
    print("✓ Warm-up done\n")

    # Time 20 inferences
    inference_times = []

    for i in range(20):
        start = time.time()
        waypoints, arrival_prob = model.predict(images, coords, step_scale=0.3)
        elapsed = time.time() - start
        inference_times.append(elapsed)

        if i == 0:
            # Show output format on first iteration
            print(f"Output format:")
            print(f"  Waypoints shape: {waypoints.shape}")
            print(f"  Waypoints (raw):")
            for j, wp in enumerate(waypoints):
                print(f"    [{j}] x={wp[0]:7.3f}, y={wp[1]:7.3f}")
            print(f"  Arrival prob: {arrival_prob:.3f}\n")

        print(f"[{i+1:2d}/20] Inference time: {elapsed*1000:6.1f}ms")

    # Statistics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    fps = 1.0 / avg_time

    print(f"\nInference timing:")
    print(f"  Average: {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms")
    print(f"  Min:     {min_time*1000:.1f}ms")
    print(f"  Max:     {max_time*1000:.1f}ms")
    print(f"  FPS:     {fps:.1f} Hz")

    # Control loop implications
    print(f"\n" + "="*70)
    print("CONTROL LOOP IMPLICATIONS")
    print("="*70)

    control_rates = [1, 5, 10, 20, 50]

    print(f"\nCan robot control loop keep up?")
    print(f"(CityWalker runs at {fps:.1f} Hz)")

    for rate in control_rates:
        control_period = 1000.0 / rate  # ms
        if avg_time * 1000 < control_period:
            status = "✓ YES"
        else:
            status = "✗ NO - Robot too fast!"
        print(f"  {rate:2d} Hz ({control_period:5.1f}ms): {status}")

    print(f"\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    print(f"\n1. Run robot control at {min(int(fps), 10)} Hz or slower")
    print(f"2. Each control loop iteration:")
    print(f"   - Get new frame")
    print(f"   - Run CityWalker inference (~{avg_time*1000:.0f}ms)")
    print(f"   - Execute waypoint as control command")
    print(f"3. Total loop time: ~{avg_time*1000 + 50:.0f}ms")

    if fps < 5:
        print(f"\n⚠ WARNING: CityWalker is slow ({fps:.1f} Hz)")
        print(f"   Robot will be sluggish. Consider:")
        print(f"   - Using GPU for inference (if not already)")
        print(f"   - Reducing image resolution")
        print(f"   - Running inference in parallel thread")


if __name__ == "__main__":
    main()
