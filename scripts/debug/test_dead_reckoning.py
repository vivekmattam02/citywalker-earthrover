"""
Test Dead Reckoning Components

Verifies that IMU gyro and wheel encoder data is being read correctly
and that the dead reckoning math is correct.

Run this BEFORE using outdoor_nav_odometry.py to catch bugs early!
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from earthrover_interface import EarthRoverInterface


def test_sensor_data_format():
    """Test that we're reading sensor data in the correct format"""
    print("="*60)
    print("TEST 1: SENSOR DATA FORMAT")
    print("="*60)

    rover = EarthRoverInterface(timeout=30.0)
    if not rover.connect():
        print("✗ Failed to connect")
        return False

    print("\nReading sensor data for 5 seconds...")
    print("\nPress Ctrl+C to stop early\n")

    try:
        for i in range(50):  # 5 seconds at 10Hz
            # Get IMU data
            imu = rover.get_imu()
            if imu is None:
                print(f"[{i}] ✗ No IMU data")
                time.sleep(0.1)
                continue

            # Parse gyro data
            gyros = imu.get('gyros', [])
            rpms = imu.get('rpms', [])
            accels = imu.get('accels', [])

            if i == 0:
                # First reading - show raw data structure
                print(f"Raw gyros data: {gyros[:20]}...")  # First 20 values
                print(f"Raw rpms data: {rpms[:20]}...")
                print(f"Raw accels data: {accels[:20]}...")
                print()

            # Parse gyro readings: data comes as [[gx, gy, gz, ts], [gx, gy, gz, ts], ...]
            gyro_readings = []
            if isinstance(gyros, list) and len(gyros) > 0:
                if isinstance(gyros[0], list):
                    # List of lists format
                    gyro_readings = [(reading[0], reading[1], reading[2], reading[3])
                                    for reading in gyros if len(reading) >= 4]
                else:
                    # Flat list format (fallback)
                    for j in range(0, len(gyros), 4):
                        if j + 3 < len(gyros):
                            gyro_readings.append((gyros[j], gyros[j+1], gyros[j+2], gyros[j+3]))

            # Parse RPM readings: data comes as [[fl, fr, rl, rr, ts], ...]
            rpm_readings = []
            if isinstance(rpms, list) and len(rpms) > 0:
                if isinstance(rpms[0], list):
                    # List of lists format
                    rpm_readings = [(reading[0], reading[1], reading[2], reading[3], reading[4])
                                   for reading in rpms if len(reading) >= 5]
                else:
                    # Flat list format (fallback)
                    for j in range(0, len(rpms), 5):
                        if j + 4 < len(rpms):
                            rpm_readings.append((rpms[j], rpms[j+1], rpms[j+2], rpms[j+3], rpms[j+4]))

            if gyro_readings:
                # Show latest gyro reading
                gx, gy, gz, ts = gyro_readings[-1]
                print(f"[{i:2d}] Gyro: gx={gx:6.2f}, gy={gy:6.2f}, gz={gz:6.2f} rad/s | ", end="")
            else:
                print(f"[{i:2d}] Gyro: NO DATA | ", end="")

            if rpm_readings:
                # Show latest RPM reading
                fl, fr, rl, rr, ts = rpm_readings[-1]
                print(f"RPM: FL={fl:5.0f}, FR={fr:5.0f}, RL={rl:5.0f}, RR={rr:5.0f}")
            else:
                print(f"RPM: NO DATA")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    print("\n" + "="*60)
    print("✓ Sensor data format test complete")
    print("="*60)
    print("\nVerify:")
    print("  1. Gyro values (gx, gy, gz) should be in rad/s")
    print("  2. When robot is still, gz should be near 0")
    print("  3. When robot is still, all RPMs should be 0")
    print("  4. Timestamps should be increasing")
    print("\nDid everything look correct? (y/n): ", end="")

    response = input().strip().lower()
    return response == 'y'


def test_gyro_integration():
    """Test gyro integration by rotating robot"""
    print("\n" + "="*60)
    print("TEST 2: GYRO INTEGRATION")
    print("="*60)
    print("\nThis test will track heading changes from gyro.")
    print("\nInstructions:")
    print("  1. Keep robot STILL for 5 seconds (heading should stay constant)")
    print("  2. Then ROTATE robot 90° clockwise")
    print("  3. Keep still again (heading should stabilize at new value)")
    print("\nPress Enter when ready...")
    input()

    rover = EarthRoverInterface(timeout=30.0)
    if not rover.connect():
        print("✗ Failed to connect")
        return False

    # Get initial heading from SDK
    _, _, initial_heading = rover.get_pose()
    print(f"\nInitial heading from SDK: {np.degrees(initial_heading):.1f}°")

    # Track heading from gyro integration
    heading = initial_heading
    last_time = time.time()

    print("\nTracking heading for 20 seconds...")
    print("Rotate the robot and watch if heading tracks correctly!\n")

    try:
        for i in range(200):  # 20 seconds at 10Hz
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Get gyro data
            imu = rover.get_imu()
            if imu is None:
                time.sleep(0.1)
                continue

            gyros = imu.get('gyros', [])

            # Parse gyro readings: [[gx, gy, gz, ts], ...]
            gyro_readings = []
            if isinstance(gyros, list) and len(gyros) > 0 and isinstance(gyros[0], list):
                gyro_readings = [(reading[0], reading[1], reading[2], reading[3])
                                for reading in gyros if len(reading) >= 4]

            if not gyro_readings:
                time.sleep(0.1)
                continue

            # Average gz (yaw rate) over all readings
            gz_values = [reading[2] for reading in gyro_readings]
            avg_gz = np.mean(gz_values)

            # Integrate to update heading
            # Positive gz = clockwise rotation = increasing heading
            delta_heading = avg_gz * dt
            heading += delta_heading

            # Normalize to [-π, π]
            heading = np.arctan2(np.sin(heading), np.cos(heading))

            # Get current SDK heading for comparison
            _, _, sdk_heading = rover.get_pose()

            # Calculate difference
            heading_diff = np.degrees(heading - sdk_heading)
            # Normalize diff to [-180, 180]
            heading_diff = ((heading_diff + 180) % 360) - 180

            print(f"\r[{i*0.1:.1f}s] "
                  f"Gyro heading: {np.degrees(heading):6.1f}° | "
                  f"SDK heading: {np.degrees(sdk_heading):6.1f}° | "
                  f"Diff: {heading_diff:5.1f}° | "
                  f"gz: {avg_gz:6.2f} rad/s   ", end="", flush=True)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    print("\n\n" + "="*60)
    print("✓ Gyro integration test complete")
    print("="*60)
    print("\nVerify:")
    print("  1. When robot is still, heading should stay constant")
    print("  2. When rotating clockwise, heading should increase")
    print("  3. Gyro heading should track SDK heading (within ~10°)")
    print("\nDid gyro integration work correctly? (y/n): ", end="")

    response = input().strip().lower()
    return response == 'y'


def test_wheel_odometry():
    """Test wheel odometry by driving forward"""
    print("\n" + "="*60)
    print("TEST 3: WHEEL ODOMETRY")
    print("="*60)
    print("\nThis test will track distance traveled from wheel encoders.")
    print("\nInstructions:")
    print("  1. Mark robot's current position")
    print("  2. Manually push robot forward ~2 meters")
    print("  3. Or use remote control to drive forward")
    print("\nPress Enter when ready...")
    input()

    rover = EarthRoverInterface(timeout=30.0)
    if not rover.connect():
        print("✗ Failed to connect")
        return False

    # Track distance
    total_distance = 0.0
    last_time = time.time()

    # Wheel parameters (VERIFY THESE!)
    WHEEL_RADIUS = 0.05  # 5cm - IS THIS CORRECT?
    WHEEL_BASE = 0.3     # 30cm - IS THIS CORRECT?

    print(f"\nUsing wheel radius: {WHEEL_RADIUS}m")
    print(f"Using wheel base: {WHEEL_BASE}m")
    print("⚠ Verify these are correct for your robot!\n")

    print("Tracking distance for 20 seconds...\n")

    try:
        for i in range(200):  # 20 seconds at 10Hz
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Get RPM data
            imu = rover.get_imu()
            if imu is None:
                time.sleep(0.1)
                continue

            rpms = imu.get('rpms', [])

            # Parse RPM readings: [[fl, fr, rl, rr, ts], ...]
            rpm_readings = []
            if isinstance(rpms, list) and len(rpms) > 0 and isinstance(rpms[0], list):
                rpm_readings = [(reading[0], reading[1], reading[2], reading[3], reading[4])
                               for reading in rpms if len(reading) >= 5]

            if not rpm_readings:
                time.sleep(0.1)
                continue

            # Get latest RPM reading
            fl, fr, rl, rr, ts = rpm_readings[-1]

            # Average left and right side RPMs
            left_rpm = (fl + rl) / 2.0
            right_rpm = (fr + rr) / 2.0

            # Convert RPM to linear velocity (m/s)
            # velocity = (RPM / 60) * (2 * π * radius)
            left_velocity = (left_rpm / 60.0) * (2 * np.pi * WHEEL_RADIUS)
            right_velocity = (right_rpm / 60.0) * (2 * np.pi * WHEEL_RADIUS)

            # Robot forward velocity (average of left and right)
            forward_velocity = (left_velocity + right_velocity) / 2.0

            # Distance traveled this step
            distance_step = forward_velocity * dt
            total_distance += distance_step

            print(f"\r[{i*0.1:.1f}s] "
                  f"RPM: L={left_rpm:5.0f}, R={right_rpm:5.0f} | "
                  f"Velocity: {forward_velocity:5.2f} m/s | "
                  f"Distance: {total_distance:5.2f}m   ", end="", flush=True)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    print("\n\n" + "="*60)
    print("✓ Wheel odometry test complete")
    print("="*60)
    print(f"\nTotal distance tracked: {total_distance:.2f}m")
    print("\nNow measure actual distance robot traveled:")
    actual_distance = float(input("Enter actual distance (meters): "))

    error = abs(total_distance - actual_distance)
    error_percent = (error / max(0.1, actual_distance)) * 100

    print(f"\nOdometry error: {error:.2f}m ({error_percent:.1f}%)")

    if error_percent < 20:
        print("✓ Odometry looks good!")
        return True
    else:
        print("✗ Odometry error is high - wheel parameters might be wrong")
        return False


def test_gps_filtering():
    """Test GPS filtering by checking for jumps"""
    print("\n" + "="*60)
    print("TEST 4: GPS FILTERING")
    print("="*60)
    print("\nThis test will monitor GPS readings and detect jumps.")
    print("\nKeep robot STILL and watch for GPS position jumps.\n")

    rover = EarthRoverInterface(timeout=30.0)
    if not rover.connect():
        print("✗ Failed to connect")
        return False

    # Get initial GPS
    lat, lon, _ = rover.get_pose()
    print(f"Initial GPS: ({lat:.8f}, {lon:.8f})\n")

    last_lat, last_lon = lat, lon
    jump_count = 0
    stable_count = 0

    print("Monitoring GPS for 30 seconds...\n")

    try:
        for i in range(30):  # 30 seconds at 1Hz
            time.sleep(1.0)

            # Get GPS
            lat, lon, _ = rover.get_pose()
            if lat is None:
                continue

            # Calculate distance from last reading
            from coordinate_utils import haversine_distance
            distance = haversine_distance(last_lat, last_lon, lat, lon)

            status = "✓ STABLE" if distance < 5.0 else "✗ JUMP"
            if distance < 5.0:
                stable_count += 1
            else:
                jump_count += 1

            print(f"[{i+1:2d}s] "
                  f"GPS: ({lat:.8f}, {lon:.8f}) | "
                  f"Move: {distance:5.2f}m | "
                  f"{status}")

            last_lat, last_lon = lat, lon

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    total = stable_count + jump_count
    stable_percent = (stable_count / max(1, total)) * 100

    print("\n" + "="*60)
    print("✓ GPS filtering test complete")
    print("="*60)
    print(f"\nStable readings: {stable_count}/{total} ({stable_percent:.0f}%)")
    print(f"Jumped readings: {jump_count}/{total}")

    if stable_percent > 70:
        print("\n✓ GPS is mostly stable - filtering should work well")
        return True
    else:
        print("\n⚠ GPS is very unstable - dead reckoning will rely heavily on IMU/wheels")
        return True  # Still ok to proceed, just means we rely less on GPS


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("DEAD RECKONING COMPONENT TESTS")
    print("="*70)
    print("\nThis will test:")
    print("  1. Sensor data format (gyro, RPM)")
    print("  2. Gyro integration for heading")
    print("  3. Wheel odometry for distance")
    print("  4. GPS filtering for jump detection")
    print("\n⚠ Run these tests BEFORE using outdoor_nav_odometry.py!")
    print("\nPress Enter to start...")
    input()

    results = {}

    # Test 1: Sensor data format
    results['sensor_format'] = test_sensor_data_format()

    if not results['sensor_format']:
        print("\n✗ Sensor data format looks wrong - fix this first!")
        return

    # Test 2: Gyro integration
    results['gyro_integration'] = test_gyro_integration()

    # Test 3: Wheel odometry
    results['wheel_odometry'] = test_wheel_odometry()

    # Test 4: GPS filtering
    results['gps_filtering'] = test_gps_filtering()

    # Summary
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✓ All tests PASSED!")
        print("\nYou can now run:")
        print("  python scripts/outdoor_nav_odometry.py --target-lat LAT --target-lon LON")
    else:
        print("\n✗ Some tests FAILED")
        print("\nFix the issues before using outdoor_nav_odometry.py")

    print("="*70)


if __name__ == "__main__":
    main()
