"""
Navigation with Depth Safety - CityWalker + Runtime Obstacle Avoidance.

How it works:
1. CityWalker predicts waypoints from camera + position
2. Depth Safety Layer verifies each waypoint is collision-free
3. If unsafe, overrides with safest available direction
4. PD Controller converts safe waypoints to velocity commands

Can run in two modes:
  - Indoor (Visual Odometry): No GPS, tracks position from camera
  - Outdoor (GPS): Uses GPS coordinates from robot

Controls:
    W = Go forward
    A = Turn left
    D = Turn right
    S = Stop
    R = Reset position
    Q = Quit

Usage:
    # Indoor with depth safety (keyboard direction control)
    python scripts/nav_with_safety.py --indoor

    # Outdoor with depth safety (GPS target)
    python scripts/nav_with_safety.py --outdoor --target-lat 40.7580 --target-lon -73.9855

    # Indoor with custom safety margin
    python scripts/nav_with_safety.py --indoor --margin 0.8

    # Dry run (no robot, test pipeline)
    python scripts/nav_with_safety.py --dry-run

Author: Vivek Mattam
"""

import sys
import os
import time
import threading
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from citywalker_wrapper import CityWalkerWrapper
from earthrover_interface import EarthRoverInterface
from pd_controller import PDController
from depth_safety import DepthSafetyLayer

# Global control
current_direction = "stop"
reset_position = False
running = True


def get_target_from_direction(direction, current_x, current_y, heading):
    """Convert direction command to target coordinates."""
    if direction == "stop":
        return current_x, current_y

    dist = 1.0
    if direction == "forward":
        angle = heading
    elif direction == "left":
        angle = heading + 0.4
    elif direction == "right":
        angle = heading - 0.4
    else:
        return current_x, current_y

    return (current_x + dist * np.cos(angle),
            current_y + dist * np.sin(angle))


def keyboard_listener():
    """Listen for keyboard input."""
    global current_direction, reset_position, running

    print("\nControls:")
    print("  W = Forward")
    print("  A = Left")
    print("  D = Right")
    print("  S = Stop")
    print("  R = Reset position")
    print("  Q = Quit")
    print("-" * 40)

    try:
        import termios
        import tty

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while running:
                char = sys.stdin.read(1).lower()
                if char == 'w':
                    current_direction = "forward"
                elif char == 'a':
                    current_direction = "left"
                elif char == 'd':
                    current_direction = "right"
                elif char == 's':
                    current_direction = "stop"
                elif char == 'r':
                    reset_position = True
                elif char == 'q':
                    current_direction = "stop"
                    running = False
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    except ImportError:
        print("(Press Enter after each command)")
        while running:
            cmd = input().strip().lower()
            if cmd in ('w', 'a', 'd', 's', 'r', 'q'):
                if cmd == 'q':
                    running = False
                elif cmd == 'r':
                    reset_position = True
                else:
                    current_direction = {'w': 'forward', 'a': 'left',
                                          'd': 'right', 's': 'stop'}[cmd]


def run_indoor(args):
    """Indoor navigation with visual odometry + depth safety."""
    global current_direction, reset_position, running

    from visual_odometry import VisualOdometry

    print("=" * 60)
    print("INDOOR NAVIGATION + DEPTH SAFETY")
    print("=" * 60)
    print(f"\nSafety margin: {args.margin}m")
    print(f"Depth model: {args.depth_model}")

    # Initialize components
    print("\n[1] Loading CityWalker model...")
    model = CityWalkerWrapper()

    print("\n[2] Loading depth safety layer...")
    safety = DepthSafetyLayer(
        model_size=args.depth_model,
        margin=args.margin,
        speed_scale=args.speed_scale
    )

    print("\n[3] Connecting to robot...")
    rover = EarthRoverInterface()
    if not rover.connect():
        print("FAILED - Is SDK server running?")
        return

    print("\n[4] Initializing Visual Odometry...")
    vo = VisualOdometry(image_size=(640, 480), scale=0.05)

    controller = PDController()
    image_buffer = []

    # Start keyboard listener
    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    print("\n[5] Starting safe navigation...")
    print("    Use W/A/S/D to control, R to reset, Q to quit")
    print("-" * 60)

    time.sleep(0.5)
    frame_count = 0
    last_status_time = time.time()

    try:
        while running:
            if reset_position:
                vo.reset()
                reset_position = False
                safety.reset_stats()

            frame = rover.get_camera_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            frame_count += 1

            # Track position
            vo_success, pos_x, pos_y, n_matches = vo.process_frame(frame)
            heading = vo.get_heading()

            # Buffer images
            image_buffer.append(frame)
            if len(image_buffer) > 5:
                image_buffer.pop(0)

            # Status every 2 seconds
            if time.time() - last_status_time > 2.0:
                stats = safety.get_stats()
                fwd_cl = stats['forward_clearance'] or 0
                print(f"\r  Pos: ({pos_x:+.2f}, {pos_y:+.2f}) | "
                      f"Dir: {current_direction:>7s} | "
                      f"Fwd: {fwd_cl:.1f}m | "
                      f"Overrides: {stats['total_overrides']} | "
                      f"Depth: {stats['last_inference_ms']:.0f}ms",
                      end="", flush=True)
                last_status_time = time.time()

            if len(image_buffer) < 5:
                time.sleep(0.1)
                continue

            if current_direction == "stop":
                rover.stop()
                time.sleep(0.1)
                continue

            # Get target
            target_x, target_y = get_target_from_direction(
                current_direction, pos_x, pos_y, heading
            )

            # Get coordinates for model
            coords = vo.get_coordinates_for_citywalker(target_x, target_y)

            # Predict waypoints
            images = np.stack(image_buffer, axis=0)
            waypoints, arrival_prob = model.predict(images, coords, step_length=1.0)

            # DEPTH SAFETY CHECK
            safe_waypoints, was_overridden = safety.check_waypoints(frame, waypoints)

            # Use safe waypoint for control
            wp = safe_waypoints[0]
            controller.reset()
            linear, angular = controller.compute(wp[0], wp[1])

            # Scale velocities
            safe_linear = max(linear * 0.5, 0.15)
            safe_angular = angular * 0.5

            # If overridden, reduce speed for caution
            if was_overridden:
                safe_linear *= 0.6
                safe_angular *= 0.8

            rover.send_control(safe_linear, safe_angular)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nCtrl+C pressed")
    finally:
        running = False
        rover.stop()
        stats = safety.get_stats()
        print(f"\n\nFinal position: ({vo.x:.2f}, {vo.y:.2f})")
        print(f"Total frames: {frame_count}")
        print(f"Safety overrides: {stats['total_overrides']}/{stats['total_checks']} "
              f"({stats['override_rate_pct']:.1f}%)")
        print("Robot stopped.")


def run_outdoor(args):
    """Outdoor navigation with GPS + depth safety."""
    global running

    from coordinate_utils import CoordinateTransformer

    print("=" * 60)
    print("OUTDOOR NAVIGATION + DEPTH SAFETY")
    print(f"Target: ({args.target_lat}, {args.target_lon})")
    print("=" * 60)
    print(f"\nSafety margin: {args.margin}m")

    # Initialize
    print("\n[1] Loading CityWalker model...")
    model = CityWalkerWrapper()

    print("\n[2] Loading depth safety layer...")
    safety = DepthSafetyLayer(
        model_size=args.depth_model,
        margin=args.margin,
        speed_scale=args.speed_scale
    )

    print("\n[3] Connecting to robot...")
    rover = EarthRoverInterface()
    if not rover.connect():
        print("FAILED - Is SDK server running?")
        return

    transformer = CoordinateTransformer(history_size=5)
    controller = PDController()
    image_buffer = []

    print("\n[4] Starting safe outdoor navigation...")
    print("    Press Ctrl+C to stop")
    print("-" * 60)

    try:
        while running:
            frame = rover.get_camera_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            lat, lon, heading = rover.get_pose()
            if lat is None:
                time.sleep(0.1)
                continue

            # Update transformer
            transformer.update(lat, lon, heading)

            # Buffer
            image_buffer.append(frame)
            if len(image_buffer) > 5:
                image_buffer.pop(0)
            if len(image_buffer) < 5 or not transformer.is_ready():
                time.sleep(0.1)
                continue

            # Check distance
            dist = transformer.get_distance_to_target(args.target_lat, args.target_lon)
            if dist < 2.0:
                rover.stop()
                print(f"\n\nARRIVED! Distance: {dist:.1f}m")
                break

            # Get model input
            coords = transformer.get_model_input(args.target_lat, args.target_lon)
            images = np.stack(image_buffer, axis=0)

            # Predict
            waypoints, arrival_prob = model.predict(images, coords, step_length=0.1)

            if arrival_prob > 0.8:
                rover.stop()
                print(f"\n\nModel says arrived! Prob: {arrival_prob:.2f}")
                break

            # DEPTH SAFETY CHECK
            safe_waypoints, was_overridden = safety.check_waypoints(frame, waypoints)

            # Control
            wp = safe_waypoints[0]
            controller.reset()
            linear, angular = controller.compute(wp[0], wp[1])

            # Scale and apply caution for overrides
            scale = 0.4 if was_overridden else 0.6
            rover.send_control(linear * scale, angular * scale)

            # Status
            stats = safety.get_stats()
            status = "OVERRIDE" if was_overridden else "SAFE"
            fwd_cl = stats['forward_clearance'] or 0
            print(f"\r  Dist: {dist:.1f}m | "
                  f"{status:>8s} | "
                  f"Fwd: {fwd_cl:.1f}m | "
                  f"Overrides: {stats['total_overrides']}",
                  end="", flush=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        running = False
        rover.stop()
        stats = safety.get_stats()
        print(f"\nSafety stats: {stats['total_overrides']}/{stats['total_checks']} overrides "
              f"({stats['override_rate_pct']:.1f}%)")


def run_dry():
    """Dry run to test the pipeline without robot."""
    print("=" * 60)
    print("DRY RUN - Testing Pipeline (no robot)")
    print("=" * 60)

    print("\n[1] Loading CityWalker model...")
    model = CityWalkerWrapper()

    print("\n[2] Loading depth safety layer...")
    try:
        safety = DepthSafetyLayer(model_size='small', margin=0.5)
    except Exception as e:
        print(f"  Could not load depth model: {e}")
        print("  Using dummy safety layer...")
        from depth_safety import DummyDepthSafety
        safety = DummyDepthSafety()

    controller = PDController()

    print("\n[3] Running 10 simulated steps...")
    for i in range(10):
        # Dummy data
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        images = np.random.randint(0, 255, (5, 480, 640, 3), dtype=np.uint8)
        coords = np.random.randn(6, 2).astype(np.float32) * 0.1

        # Predict
        waypoints, arrival_prob = model.predict(images, coords, step_length=0.1)

        # Safety check
        safe_waypoints, was_overridden = safety.check_waypoints(frame, waypoints)

        # Control
        wp = safe_waypoints[0]
        linear, angular = controller.compute(wp[0], wp[1])

        status = "OVERRIDE" if was_overridden else "OK"
        print(f"  Step {i+1}: wp=({wp[0]:+.3f}, {wp[1]:+.3f}) | "
              f"vel=({linear:+.3f}, {angular:+.3f}) | "
              f"{status}")

    stats = safety.get_stats()
    print(f"\nDone! Overrides: {stats['total_overrides']}/{stats['total_checks']}")
    print("Pipeline works correctly.")


def main():
    global running

    parser = argparse.ArgumentParser(description="Navigation with Depth Safety")

    # Mode
    parser.add_argument('--indoor', action='store_true',
                        help='Indoor mode (Visual Odometry + keyboard)')
    parser.add_argument('--outdoor', action='store_true',
                        help='Outdoor mode (GPS target)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Test pipeline without robot')

    # Outdoor target
    parser.add_argument('--target-lat', type=float, default=None,
                        help='Target latitude (outdoor mode)')
    parser.add_argument('--target-lon', type=float, default=None,
                        help='Target longitude (outdoor mode)')

    # Safety parameters
    parser.add_argument('--margin', type=float, default=0.5,
                        help='Safety margin in meters (default 0.5)')
    parser.add_argument('--depth-model', type=str, default='small',
                        choices=['small', 'base', 'large'],
                        help='Depth model size (default small)')
    parser.add_argument('--speed-scale', type=float, default=0.5,
                        help='Speed scale for safe override waypoints (default 0.5)')

    args = parser.parse_args()

    if args.dry_run:
        run_dry()
    elif args.outdoor:
        if args.target_lat is None or args.target_lon is None:
            print("ERROR: --outdoor requires --target-lat and --target-lon")
            return
        run_outdoor(args)
    elif args.indoor:
        run_indoor(args)
    else:
        # Default: dry run
        print("No mode specified. Use --indoor, --outdoor, or --dry-run")
        print("\nExamples:")
        print("  python scripts/nav_with_safety.py --indoor")
        print("  python scripts/nav_with_safety.py --outdoor --target-lat 40.75 --target-lon -73.98")
        print("  python scripts/nav_with_safety.py --dry-run")
        print("\nRunning dry run by default...\n")
        run_dry()


if __name__ == "__main__":
    main()
