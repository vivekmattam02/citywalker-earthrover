"""
Indoor Navigation - Visual obstacle avoidance without GPS.

How it works:
1. You set a DIRECTION (forward, left, right) with keyboard
2. Visual Odometry tracks robot position from camera
3. CityWalker uses CAMERA + REAL POSITION to navigate safely
4. Model avoids obstacles automatically based on what it sees

Controls:
    W = Go forward
    A = Turn left
    D = Turn right
    S = Stop
    R = Reset position to origin
    Q = Quit

Usage:
    python scripts/indoor_nav.py
"""

import sys
import os
import time
import threading
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from citywalker_wrapper import CityWalkerWrapper
from pd_controller import PDController
from visual_odometry import VisualOdometry
from earthrover_interface import EarthRoverInterface

# Global control
current_direction = "stop"  # forward, left, right, stop
reset_position = False
running = True


def get_target_from_direction(direction: str, current_x: float, current_y: float, heading: float):
    """
    Convert direction command to absolute target coordinates.

    Args:
        direction: "forward", "left", "right", "backward", or "stop"
        current_x, current_y: Current position in meters
        heading: Current heading in radians

    Returns:
        target_x, target_y: Absolute target position
    """
    if direction == "stop":
        return current_x, current_y

    # Distance to target
    dist = 1.0  # 1 meter ahead

    if direction == "forward":
        angle = heading
    elif direction == "left":
        angle = heading + 1.2  # ~70 degrees left (stronger turn)
    elif direction == "right":
        angle = heading - 1.2  # ~70 degrees right (stronger turn)
    elif direction == "backward":
        angle = heading + np.pi  # 180 degrees - go backward
    else:
        return current_x, current_y

    target_x = current_x + dist * np.cos(angle)
    target_y = current_y + dist * np.sin(angle)

    return target_x, target_y


def keyboard_listener():
    """Listen for keyboard input in a separate thread."""
    global current_direction, reset_position, running

    print("\nControls:")
    print("  W = Forward")
    print("  A = Turn Left")
    print("  D = Turn Right")
    print("  S = Backward")
    print("  X = Stop")
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
                    print("\r>> FORWARD      ", end="", flush=True)
                elif char == 'a':
                    current_direction = "left"
                    print("\r>> LEFT         ", end="", flush=True)
                elif char == 'd':
                    current_direction = "right"
                    print("\r>> RIGHT        ", end="", flush=True)
                elif char == 's':
                    current_direction = "backward"
                    print("\r>> BACKWARD     ", end="", flush=True)
                elif char == 'x' or char == ' ':
                    current_direction = "stop"
                    print("\r>> STOP         ", end="", flush=True)
                elif char == 'r':
                    reset_position = True
                    print("\r>> RESET        ", end="", flush=True)
                elif char == 'q':
                    current_direction = "stop"
                    running = False
                    print("\r>> QUIT         ", end="", flush=True)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    except ImportError:
        # Windows fallback
        print("(Press Enter after each command)")
        while running:
            cmd = input().strip().lower()
            if cmd == 'w':
                current_direction = "forward"
            elif cmd == 'a':
                current_direction = "left"
            elif cmd == 'd':
                current_direction = "right"
            elif cmd == 's':
                current_direction = "backward"
            elif cmd == 'x':
                current_direction = "stop"
            elif cmd == 'r':
                reset_position = True
            elif cmd == 'q':
                running = False


def main():
    global current_direction, reset_position, running

    print("=" * 60)
    print("INDOOR NAVIGATION - Visual Odometry + CityWalker")
    print("=" * 60)
    print("\nHow it works:")
    print("  - Visual Odometry tracks REAL position from camera")
    print("  - You set DIRECTION (W/A/S/D keys)")
    print("  - CityWalker navigates safely using real coordinates")
    print("=" * 60)

    # Initialize
    print("\n[1] Loading CityWalker model...")
    model = CityWalkerWrapper()

    print("\n[2] Connecting to robot...")
    print("    Make sure:")
    print("    1. SDK server is running (hypercorn main:app)")
    print("    2. Browser is open at http://localhost:8000/sdk")
    print("    3. Wait 5 seconds for data to flow")
    rover = EarthRoverInterface(timeout=10.0)
    if not rover.connect():
        print("FAILED - Check SDK server and browser")
        return

    print("\n[3] Initializing Visual Odometry...")
    vo = VisualOdometry(image_size=(640, 480), scale=0.05)
    print("    Scale factor: 0.05 (tune if needed)")

    # Tuned gains for smooth, slower movement
    controller = PDController(
        kp_linear=0.5,    # Gentler forward response
        kd_linear=0.05,
        kp_angular=1.0,   # Responsive turning
        kd_angular=0.1,
        max_linear=0.3,   # SLOWER max speed (was 0.5)
        max_angular=0.4   # SLOWER turning (was 0.6)
    )

    # Image buffer
    image_buffer = []

    # Start keyboard listener
    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    print("\n[4] Starting navigation...")
    print("    Use W/A/S/D to control, R to reset, Q to quit")
    print("-" * 60)

    time.sleep(0.5)  # Let keyboard thread start

    frame_count = 0
    last_status_time = time.time()

    try:
        while running:
            # Check for position reset
            if reset_position:
                vo.reset()
                reset_position = False
                print("\r>> Position reset to (0, 0)")

            # Get camera frame
            frame = rover.get_camera_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            frame_count += 1

            # Update visual odometry (track position from camera)
            vo_success, pos_x, pos_y, n_matches = vo.process_frame(frame)
            heading = vo.get_heading()

            # Update buffer
            image_buffer.append(frame)
            if len(image_buffer) > 5:
                image_buffer.pop(0)

            # Print status every 2 seconds
            if time.time() - last_status_time > 2.0:
                print(f"\r   Pos: ({pos_x:+.2f}, {pos_y:+.2f})  "
                      f"Heading: {np.degrees(heading):+.0f}deg  "
                      f"Matches: {n_matches}  "
                      f"Dir: {current_direction}     ", end="", flush=True)
                last_status_time = time.time()

            # Need 5 frames
            if len(image_buffer) < 5:
                time.sleep(0.1)
                continue

            # Handle stop
            if current_direction == "stop":
                rover.stop()
                time.sleep(0.1)
                continue

            # Get target based on direction and current position
            target_x, target_y = get_target_from_direction(
                current_direction, pos_x, pos_y, heading
            )

            # Get coordinates for CityWalker (uses real position history)
            coords = vo.get_coordinates_for_citywalker(target_x, target_y)
            
            # DEBUG: Show target in robot frame (last row of coords is target)
            target_robot_frame = coords[5]  # Index 5 is the target

            # Run model with proper normalization (step_scale=0.3m from training data)
            images = np.stack(image_buffer, axis=0)
            waypoints, arrival_prob = model.predict(images, coords, step_scale=0.3)

            # Get first waypoint (in robot frame: X=forward, Y=left)
            wp = waypoints[0]
            
            # Compute velocities from waypoint
            linear, angular = controller.compute(wp[0], wp[1])
            
            # DEBUG: Print everything
            print(f"\r   Dir:{current_direction:7s} Target:({target_robot_frame[0]:+.2f},{target_robot_frame[1]:+.2f}) "
                  f"WP:({wp[0]:+.2f},{wp[1]:+.2f}) -> L:{linear:+.2f} A:{angular:+.2f}   ", end="", flush=True)

            # Send to robot
            rover.send_control(linear, angular)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nCtrl+C pressed")
    finally:
        running = False
        rover.stop()
        print(f"\n\nFinal position: ({vo.x:.2f}, {vo.y:.2f})")
        print(f"Total frames: {frame_count}")
        print("Robot stopped. Goodbye!")


if __name__ == "__main__":
    main()
