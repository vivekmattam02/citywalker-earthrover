"""
Test Depth Estimation and Safety Checking

Tests depth estimation on the robot's camera feed and shows:
- Depth map visualization
- Polar clearance per direction
- Safety checking for waypoints

Can also be used as a runtime safety layer on top of CityWalker.

USAGE:
    # Test with robot camera (needs SDK running)
    python scripts/test_depth.py --live

    # Test with a saved image
    python scripts/test_depth.py --image path/to/image.jpg

    # Test with dummy data (no robot needed)
    python scripts/test_depth.py --dummy

    # Run as safety layer with CityWalker
    python scripts/test_depth.py --safe-nav --target 2.0 0.0

Author: Vivek Mattam
"""

import sys
import os
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_dummy():
    """Test depth estimation with dummy data."""
    from depth_estimator import DepthEstimator

    print("=" * 60)
    print("DEPTH ESTIMATION TEST (Dummy Data)")
    print("=" * 60)

    # Create estimator
    estimator = DepthEstimator(model_size='small')

    # Create dummy image
    print("\n[1] Creating dummy image (480x640)...")
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Estimate depth
    print("[2] Estimating depth...")
    start = time.time()
    depth = estimator.estimate(image)
    elapsed = (time.time() - start) * 1000
    print(f"  Depth shape: {depth.shape}")
    print(f"  Depth range: [{depth.min():.2f}, {depth.max():.2f}] meters")
    print(f"  Inference time: {elapsed:.1f} ms")

    # Compute polar clearance
    print("\n[3] Computing polar clearance (32 bins)...")
    clearance, bin_centers = estimator.get_polar_clearance(depth)
    print(f"  Clearance shape: {clearance.shape}")
    print(f"  Direction range: [{np.degrees(bin_centers[0]):.1f}°, {np.degrees(bin_centers[-1]):.1f}°]")

    # Print clearance per direction
    print("\n  Clearance per direction:")
    print("  " + "-" * 50)
    for i in range(0, len(bin_centers), 4):  # Every 4th bin
        angle = np.degrees(bin_centers[i])
        cl = clearance[i]
        bar = "█" * int(min(cl, 10) * 3)
        safe = "SAFE" if cl >= 0.5 else "DANGER"
        print(f"  {angle:+6.1f}°: {cl:5.2f}m {bar} [{safe}]")

    # Test waypoint safety
    print("\n[4] Testing waypoint safety...")
    test_waypoints = [
        (1.0, 0.0, "straight ahead"),
        (1.0, 1.0, "ahead-left"),
        (1.0, -1.0, "ahead-right"),
        (0.0, 1.0, "pure left"),
    ]

    for x, y, desc in test_waypoints:
        safe, cl = estimator.is_waypoint_safe(
            np.array([x, y]), clearance, bin_centers, margin=0.5)
        status = "SAFE" if safe else "UNSAFE"
        print(f"  ({x:+.1f}, {y:+.1f}) [{desc:>12s}]: {status} (clearance={cl:.2f}m)")

    # Find safest direction
    print("\n[5] Finding safest direction...")
    best_angle, best_cl = estimator.get_safe_direction(clearance, bin_centers)
    print(f"  Best direction: {np.degrees(best_angle):.1f}° with {best_cl:.2f}m clearance")

    print("\nTest complete!")


def test_with_image(image_path):
    """Test depth estimation with a saved image."""
    import cv2
    from depth_estimator import DepthEstimator

    print("=" * 60)
    print(f"DEPTH ESTIMATION TEST (Image: {image_path})")
    print("=" * 60)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot load image {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {img_rgb.shape}")

    # Estimate depth
    estimator = DepthEstimator(model_size='small')
    depth = estimator.estimate(img_rgb)
    print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}] meters")

    # Compute clearance
    clearance, bin_centers = estimator.get_polar_clearance(depth)

    # Print results
    print("\nPolar Clearance:")
    print("-" * 50)
    for i in range(len(bin_centers)):
        angle = np.degrees(bin_centers[i])
        cl = clearance[i]
        bar = "█" * int(min(cl, 10) * 3)
        safe = "✓" if cl >= 0.5 else "✗"
        print(f"  {angle:+6.1f}°: {cl:5.2f}m {bar} {safe}")

    # Save depth map
    output_path = image_path.replace('.', '_depth.')
    np.save(output_path.replace(output_path.split('.')[-1], 'npy'), depth)
    print(f"\nDepth saved to: {output_path.replace(output_path.split('.')[-1], 'npy')}")


def test_live():
    """Test depth estimation with live robot camera."""
    from depth_estimator import DepthEstimator
    from earthrover_interface import EarthRoverInterface

    print("=" * 60)
    print("DEPTH ESTIMATION TEST (Live Robot Camera)")
    print("=" * 60)

    # Connect to robot
    rover = EarthRoverInterface()
    if not rover.connect():
        print("ERROR: Cannot connect to robot")
        return

    # Load depth estimator
    estimator = DepthEstimator(model_size='small')

    print("\nCapturing frames and estimating depth...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            # Get frame
            frame = rover.get_camera_frame()
            if frame is None:
                print("No frame received")
                time.sleep(1)
                continue

            # Estimate depth
            start = time.time()
            depth = estimator.estimate(frame)
            elapsed = (time.time() - start) * 1000

            # Compute clearance
            clearance, bin_centers = estimator.get_polar_clearance(depth)

            # Find safest and most dangerous directions
            safest_idx = np.argmax(clearance)
            danger_idx = np.argmin(clearance)

            # Print summary
            print(f"\rDepth: [{depth.min():.1f}-{depth.max():.1f}m] | "
                  f"Min clear: {clearance.min():.2f}m @ {np.degrees(bin_centers[danger_idx]):.0f}° | "
                  f"Max clear: {clearance.max():.2f}m @ {np.degrees(bin_centers[safest_idx]):.0f}° | "
                  f"{elapsed:.0f}ms",
                  end="", flush=True)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nStopped.")


def safe_navigation(target_x, target_y):
    """
    Run CityWalker with runtime depth safety checking.

    This is an alternative to DBR training - uses depth at inference
    to verify waypoints are safe before executing them.
    """
    from depth_estimator import DepthEstimator
    from citywalker_wrapper import CityWalkerWrapper
    from earthrover_interface import EarthRoverInterface
    from visual_odometry import VisualOdometry
    from pd_controller import PDController

    print("=" * 60)
    print("SAFE NAVIGATION (CityWalker + Depth Safety)")
    print(f"Target: ({target_x}, {target_y})")
    print("=" * 60)

    # Initialize
    model = CityWalkerWrapper()
    estimator = DepthEstimator(model_size='small')
    rover = EarthRoverInterface()
    vo = VisualOdometry(scale=0.05)
    controller = PDController()

    if not rover.connect():
        print("ERROR: Cannot connect to robot")
        return

    image_buffer = []
    safety_overrides = 0

    print("\nNavigating with depth safety checks...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            # Get frame
            frame = rover.get_camera_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # Track position
            success, pos_x, pos_y, n_matches = vo.process_frame(frame)

            # Buffer images
            image_buffer.append(frame)
            if len(image_buffer) > 5:
                image_buffer.pop(0)
            if len(image_buffer) < 5:
                continue

            # Get coordinates for model
            coords = vo.get_coordinates_for_citywalker(target_x, target_y)

            # Predict waypoints
            images = np.stack(image_buffer, axis=0)
            waypoints, arrival_prob = model.predict(images, coords)

            # SAFETY CHECK: Verify first waypoint is safe
            depth = estimator.estimate(frame)
            clearance, bin_centers = estimator.get_polar_clearance(depth)

            wp = waypoints[0]
            safe, wp_clearance = estimator.is_waypoint_safe(
                wp, clearance, bin_centers, margin=0.5)

            if safe:
                # Waypoint is safe, use it
                linear, angular = controller.compute(wp[0], wp[1])
            else:
                # UNSAFE! Override with safe direction
                safety_overrides += 1
                best_angle, best_cl = estimator.get_safe_direction(
                    clearance, bin_centers, margin=0.5)

                # Create safe waypoint in best direction
                safe_wp_x = np.cos(best_angle) * 0.5
                safe_wp_y = np.sin(best_angle) * 0.5
                linear, angular = controller.compute(safe_wp_x, safe_wp_y)

                print(f"\n  [SAFETY OVERRIDE #{safety_overrides}] "
                      f"Clearance={wp_clearance:.2f}m < 0.5m | "
                      f"Redirecting to {np.degrees(best_angle):.0f}°")

            # Send command
            rover.send_control(linear * 0.4, angular * 0.4)

            # Check arrival
            dist = np.sqrt((target_x - pos_x)**2 + (target_y - pos_y)**2)
            if dist < 0.3 or arrival_prob > 0.7:
                rover.stop()
                print(f"\n\nARRIVED! Distance: {dist:.2f}m")
                print(f"Safety overrides: {safety_overrides}")
                break

            # Status
            print(f"\rPos: ({pos_x:.2f}, {pos_y:.2f}) | "
                  f"Dist: {dist:.2f}m | "
                  f"Clear: {wp_clearance:.2f}m | "
                  f"{'SAFE' if safe else 'OVERRIDE'} | "
                  f"Overrides: {safety_overrides}",
                  end="", flush=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        rover.stop()
        print(f"\n\nStopped. Safety overrides: {safety_overrides}")


def main():
    parser = argparse.ArgumentParser(description="Test Depth Estimation & Safety")

    parser.add_argument('--dummy', action='store_true',
                        help='Test with dummy data (no robot/image needed)')
    parser.add_argument('--image', type=str, default=None,
                        help='Test with a saved image')
    parser.add_argument('--live', action='store_true',
                        help='Test with live robot camera')
    parser.add_argument('--safe-nav', action='store_true',
                        help='Run navigation with depth safety')
    parser.add_argument('--target', nargs=2, type=float, default=[2.0, 0.0],
                        help='Target position for safe-nav (x y in meters)')

    args = parser.parse_args()

    if args.safe_nav:
        safe_navigation(args.target[0], args.target[1])
    elif args.live:
        test_live()
    elif args.image:
        test_with_image(args.image)
    else:
        test_dummy()


if __name__ == "__main__":
    main()
