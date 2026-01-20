"""
Live test - Run CityWalker on live camera without GPS navigation.

This tests:
1. Camera capture speed
2. Model inference speed
3. Motor command generation

Usage:
    python scripts/test_live.py --no-move    # Just predict, don't move
    python scripts/test_live.py              # Predict and move robot
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


def main(move_robot=False, num_steps=20):
    print("=" * 60)
    print("LIVE TEST - CityWalker on EarthRover")
    print("=" * 60)

    # Initialize components
    print("\n[1] Loading model...")
    start = time.time()
    model = CityWalkerWrapper()
    print(f"    Model loaded in {time.time() - start:.1f}s")

    print("\n[2] Connecting to robot...")
    rover = EarthRoverInterface()
    if not rover.connect():
        print("    FAILED - Is SDK server running?")
        return

    controller = PDController()

    # Image buffer (need 5 frames)
    image_buffer = []

    # Fake coordinates (since no GPS)
    # Pretend robot is at origin, target is 1m ahead (closer target)
    fake_coords = np.array([
        [-0.4, 0.0],  # past 5: was 0.4m behind
        [-0.3, 0.0],  # past 4: was 0.3m behind
        [-0.2, 0.0],  # past 3: was 0.2m behind
        [-0.1, 0.0],  # past 2: was 0.1m behind
        [0.0, 0.0],   # past 1 (current position)
        [1.0, 0.0],   # target: 1m forward
    ], dtype=np.float32)

    print("\n[3] Starting live test...")
    print(f"    Move robot: {move_robot}")
    print(f"    Steps: {num_steps}")
    print("-" * 60)

    inference_times = []

    try:
        for step in range(num_steps):
            step_start = time.time()

            # Get camera frame
            t0 = time.time()
            frame = rover.get_camera_frame()
            camera_time = time.time() - t0

            if frame is None:
                print(f"Step {step+1}: No camera frame")
                continue

            # Add to buffer
            image_buffer.append(frame)
            if len(image_buffer) > 5:
                image_buffer.pop(0)

            # Need 5 frames
            if len(image_buffer) < 5:
                print(f"Step {step+1}: Buffering... ({len(image_buffer)}/5 frames)")
                time.sleep(0.1)
                continue

            # Stack images
            images = np.stack(image_buffer, axis=0)

            # Run model
            t0 = time.time()
            waypoints, arrival_prob = model.predict(images, fake_coords, step_length=1.0)
            inference_time = time.time() - t0
            inference_times.append(inference_time)

            # Get first waypoint
            wp = waypoints[0]

            # Compute motor commands
            controller.reset()
            linear, angular = controller.compute(wp[0], wp[1])

            total_time = time.time() - step_start
            fps = 1.0 / total_time if total_time > 0 else 0

            print(f"Step {step+1:2d}: waypoint=({wp[0]:+.3f}, {wp[1]:+.3f})  "
                  f"vel=({linear:+.2f}, {angular:+.2f})  "
                  f"arrived={arrival_prob:.2f}  "
                  f"infer={inference_time*1000:.0f}ms  "
                  f"fps={fps:.1f}")

            # Send to robot if enabled
            if move_robot:
                # Scale down for safety (but add minimum forward motion)
                safe_linear = max(linear * 0.5, 0.15)  # At least 15% forward
                safe_angular = angular * 0.5
                rover.send_control(safe_linear, safe_angular)

            # Small delay to not overwhelm
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        # Always stop robot
        rover.stop()
        print("\nRobot stopped.")

    # Stats
    if inference_times:
        avg_time = np.mean(inference_times) * 1000
        min_time = np.min(inference_times) * 1000
        max_time = np.max(inference_times) * 1000
        print("\n" + "=" * 60)
        print("PERFORMANCE STATS")
        print("=" * 60)
        print(f"Inference time: avg={avg_time:.0f}ms, min={min_time:.0f}ms, max={max_time:.0f}ms")
        print(f"Possible FPS: {1000/avg_time:.1f}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-move", action="store_true", help="Don't move robot, just predict")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps")
    args = parser.parse_args()

    main(move_robot=not args.no_move, num_steps=args.steps)
