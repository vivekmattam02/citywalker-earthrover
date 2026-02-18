"""
Sensor Diagnostic Tool

This will help identify what's broken:
1. Is GPS updating? (take readings 5 seconds apart)
2. Are wheel encoders working? (check for non-zero RPMs)
3. Is heading changing? (SDK fusion working?)
4. Is camera working? (getting frames?)

Run this and move the robot manually to see what updates.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from earthrover_interface import EarthRoverInterface
from coordinate_utils import haversine_distance


def main():
    print("="*70)
    print("SENSOR DIAGNOSTIC")
    print("="*70)
    print("\nThis will check what sensors are working.\n")
    print("Instructions:")
    print("  1. Keep robot stationary for 10 seconds")
    print("  2. Then DRIVE ROBOT FORWARD for 5 seconds")
    print("  3. Then TURN ROBOT for 5 seconds")
    print("  4. We'll see which sensors detect the movement\n")

    input("Press ENTER when ready to start...")

    # Connect
    print("\nConnecting to robot...")
    rover = EarthRoverInterface(timeout=30.0)
    if not rover.connect():
        print("✗ Failed to connect")
        return

    print("✓ Connected\n")

    # Get initial readings
    print("="*70)
    print("INITIAL READINGS (robot should be stationary)")
    print("="*70)

    initial_lat, initial_lon, initial_heading = rover.get_pose()
    if initial_lat is None:
        print("✗ No GPS data!")
        return

    print(f"\nGPS Position:")
    print(f"  Lat: {initial_lat:.8f}")
    print(f"  Lon: {initial_lon:.8f}")
    print(f"  Heading: {np.degrees(initial_heading):.1f}°")

    # Check camera
    frame = rover.get_camera_frame()
    if frame is not None:
        print(f"\nCamera: ✓ Working (shape: {frame.shape})")
    else:
        print("\nCamera: ✗ No frames")

    # Check IMU/RPM
    imu_data = rover.get_imu()
    if imu_data:
        rpm_data = imu_data.get('rpms', [])
        gyro_data = imu_data.get('gyroscope', [])
        print(f"\nIMU Data:")
        print(f"  RPM readings: {len(rpm_data)} samples")
        print(f"  Gyro readings: {len(gyro_data)} samples")

        if len(rpm_data) > 0:
            print(f"  RPM sample: {rpm_data[0]}")
        if len(gyro_data) > 0:
            print(f"  Gyro sample: {gyro_data[0]}")
    else:
        print("\nIMU Data: ✗ No data")

    # Stationary test - 10 seconds
    print("\n" + "="*70)
    print("TEST 1: STATIONARY (10 seconds)")
    print("="*70)
    print("Keep robot still. Checking if sensors drift...\n")

    time.sleep(2)

    stationary_readings = []
    for i in range(10):
        lat, lon, heading = rover.get_pose()
        imu = rover.get_imu()

        rpm_count = 0
        if imu and 'rpms' in imu:
            rpm_count = len(imu['rpms'])

        stationary_readings.append((lat, lon, heading))

        dist_from_start = haversine_distance(initial_lat, initial_lon, lat, lon)
        print(f"[{i+1:2d}s] GPS dist: {dist_from_start:6.2f}m | "
              f"H: {np.degrees(heading):6.1f}° | "
              f"RPM samples: {rpm_count}")

        time.sleep(1)

    # Calculate drift
    final_lat, final_lon, final_heading = stationary_readings[-1]
    stationary_drift = haversine_distance(initial_lat, initial_lon, final_lat, final_lon)
    heading_drift = abs(np.degrees(final_heading - initial_heading))

    print(f"\nStationary Results:")
    print(f"  GPS drift: {stationary_drift:.2f}m (should be <5m)")
    print(f"  Heading drift: {heading_drift:.1f}° (should be <5°)")

    # Movement test
    print("\n" + "="*70)
    print("TEST 2: DRIVE FORWARD (5 seconds)")
    print("="*70)
    print("DRIVE ROBOT FORWARD NOW!")
    print("(Use SDK browser: http://localhost:8000/sdk)\n")

    time.sleep(2)

    drive_start_lat, drive_start_lon, drive_start_heading = rover.get_pose()

    for i in range(5):
        lat, lon, heading = rover.get_pose()
        imu = rover.get_imu()

        # Check for non-zero RPMs
        max_rpm = 0
        rpm_count = 0
        if imu and 'rpms' in imu:
            rpm_data = imu['rpms']
            rpm_count = len(rpm_data)
            if isinstance(rpm_data, list) and len(rpm_data) > 0:
                if isinstance(rpm_data[0], list):
                    # [[fl, fr, rl, rr, ts], ...]
                    for reading in rpm_data:
                        if len(reading) >= 4:
                            max_rpm = max(max_rpm, abs(reading[0]), abs(reading[1]),
                                        abs(reading[2]), abs(reading[3]))

        dist_moved = haversine_distance(drive_start_lat, drive_start_lon, lat, lon)
        print(f"[{i+1}s] Moved: {dist_moved:6.2f}m | "
              f"H: {np.degrees(heading):6.1f}° | "
              f"Max RPM: {max_rpm:6.1f} | "
              f"RPM samples: {rpm_count}")

        time.sleep(1)

    drive_end_lat, drive_end_lon, drive_end_heading = rover.get_pose()
    total_drive_dist = haversine_distance(drive_start_lat, drive_start_lon,
                                          drive_end_lat, drive_end_lon)

    print(f"\nDrive Results:")
    print(f"  GPS distance moved: {total_drive_dist:.2f}m")
    print(f"  (Should be >0.5m if you drove forward)")

    # Turn test
    print("\n" + "="*70)
    print("TEST 3: TURN ROBOT (5 seconds)")
    print("="*70)
    print("TURN ROBOT NOW! (spin in place)\n")

    time.sleep(2)

    turn_start_heading = rover.get_pose()[2]

    for i in range(5):
        lat, lon, heading = rover.get_pose()

        heading_change = abs(np.degrees(heading - turn_start_heading))
        print(f"[{i+1}s] Heading change: {heading_change:6.1f}°")

        time.sleep(1)

    turn_end_heading = rover.get_pose()[2]
    total_heading_change = abs(np.degrees(turn_end_heading - turn_start_heading))

    print(f"\nTurn Results:")
    print(f"  Heading changed: {total_heading_change:.1f}°")
    print(f"  (Should be >30° if you turned)")

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    print(f"\n✓ GPS Position: {'WORKING' if initial_lat is not None else 'BROKEN'}")

    if stationary_drift < 5.0:
        print(f"✓ GPS Stable when stationary: YES (drift {stationary_drift:.2f}m)")
    else:
        print(f"✗ GPS Stable when stationary: NO (drift {stationary_drift:.2f}m - too high!)")

    if total_drive_dist > 0.5:
        print(f"✓ GPS detects movement: YES ({total_drive_dist:.2f}m)")
    else:
        print(f"✗ GPS detects movement: NO ({total_drive_dist:.2f}m - should be >0.5m)")
        print(f"  → Either GPS is frozen OR you didn't drive the robot")

    if total_heading_change > 30:
        print(f"✓ Heading updates: YES ({total_heading_change:.1f}°)")
    else:
        print(f"✗ Heading updates: NO ({total_heading_change:.1f}° - should be >30°)")
        print(f"  → Either heading is frozen OR you didn't turn the robot")

    if max_rpm > 1.0:
        print(f"✓ Wheel encoders: WORKING (max RPM: {max_rpm:.1f})")
    else:
        print(f"✗ Wheel encoders: NOT WORKING (max RPM: {max_rpm:.1f})")
        print(f"  → RPMs always zero - encoders broken or robot not moving")

    if frame is not None:
        print(f"✓ Camera: WORKING")
    else:
        print(f"✗ Camera: NOT WORKING")

    print("\n" + "="*70)
    print("\nNext steps based on results:")
    print("  - If GPS not detecting movement → GPS frozen or signal lost")
    print("  - If heading not updating → SDK heading fusion broken")
    print("  - If RPMs always zero → Wheel encoders broken OR motors not running")
    print("  - If you didn't drive → Use SDK browser to drive manually")
    print("\n")


if __name__ == "__main__":
    main()
