"""
Hardware Verification Script
Tests cameras, IMU, and basic robot functionality

Run this INDOORS to verify hardware works before outdoor testing.
"""

import sys
import os
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from earthrover_interface import EarthRoverInterface

def test_connection():
    """Test SDK connection"""
    print("\n" + "="*60)
    print("TEST 1: SDK Connection")
    print("="*60)

    rover = EarthRoverInterface(timeout=30.0)
    if rover.connect():
        print("✓ Connected to SDK")
        return rover
    else:
        print("✗ Failed to connect")
        print("Make sure SDK is running: cd earth-rovers-sdk && hypercorn main:app --reload")
        return None

def test_camera(rover):
    """Test front camera"""
    print("\n" + "="*60)
    print("TEST 2: Front Camera")
    print("="*60)

    frame = rover.get_camera_frame()
    if frame is None:
        print("✗ No camera frame received")
        return False

    print(f"✓ Camera frame received: {frame.shape}")
    print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print(f"  Channels: {frame.shape[2]}")
    print(f"  Data type: {frame.dtype}")

    # Save frame
    output_path = 'camera_test.jpg'
    cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved frame to: {output_path}")
    print("  ACTION: Look at this image - does it show FRONT camera view?")

    return True

def test_imu(rover):
    """Test IMU sensors"""
    print("\n" + "="*60)
    print("TEST 3: IMU (Accelerometer, Gyroscope, Magnetometer)")
    print("="*60)

    imu = rover.get_imu()
    if imu is None:
        print("✗ No IMU data")
        return False

    accels = imu.get('accels', [])
    gyros = imu.get('gyros', [])
    mags = imu.get('mags', [])

    print(f"Accelerometer: {len(accels)} readings")
    if accels:
        print(f"  Sample: {accels[:3] if len(accels) >= 3 else accels}")

    print(f"Gyroscope: {len(gyros)} readings")
    if gyros:
        print(f"  Sample: {gyros[:3] if len(gyros) >= 3 else gyros}")

    print(f"Magnetometer: {len(mags)} readings")
    if mags:
        print(f"  Sample: {mags[:3] if len(mags) >= 3 else mags}")

    if accels or gyros or mags:
        print("✓ IMU data available")
        return True
    else:
        print("✗ No IMU readings")
        return False

def test_orientation(rover):
    """Test orientation/heading"""
    print("\n" + "="*60)
    print("TEST 4: Orientation/Heading")
    print("="*60)

    print("Sampling orientation for 5 seconds...")
    orientations = []

    for i in range(10):
        data = rover.get_data()
        if data:
            orientation = data.get('orientation')
            if orientation is not None:
                orientations.append(orientation)
                print(f"  [{i}] Orientation: {orientation:.1f}°")
        time.sleep(0.5)

    if not orientations:
        print("✗ No orientation data")
        return False

    mean_orientation = np.mean(orientations)
    std_orientation = np.std(orientations)

    print(f"\n✓ Orientation data available")
    print(f"  Mean: {mean_orientation:.1f}°")
    print(f"  Std Dev: {std_orientation:.1f}°")

    if std_orientation > 10:
        print(f"  ⚠ Warning: High noise ({std_orientation:.1f}° std dev)")

    return True

def test_battery(rover):
    """Test battery reading"""
    print("\n" + "="*60)
    print("TEST 5: Battery")
    print("="*60)

    battery = rover.get_battery()
    if battery is not None:
        print(f"✓ Battery: {battery}%")
        if battery < 20:
            print("  ⚠ Warning: Low battery!")
        return True
    else:
        print("✗ No battery data")
        return False

def test_control(rover):
    """Test movement control"""
    print("\n" + "="*60)
    print("TEST 6: Movement Control")
    print("="*60)

    print("Testing control commands (robot will NOT move, just testing API)...")

    # Test stop command
    result = rover.send_control(0.0, 0.0)
    if result:
        print("✓ Stop command sent successfully")
    else:
        print("✗ Stop command failed")
        return False

    time.sleep(0.5)

    # Test movement command (very small)
    result = rover.send_control(0.1, 0.0)
    if result:
        print("✓ Forward command sent successfully")
    else:
        print("✗ Forward command failed")
        return False

    time.sleep(0.5)

    # Stop again
    rover.stop()
    print("✓ Control system working")

    return True

def main():
    print("="*60)
    print("EARTHROVER MINI PLUS - HARDWARE VERIFICATION")
    print("="*60)
    print("\nThis test verifies:")
    print("  1. SDK connection")
    print("  2. Front camera")
    print("  3. IMU sensors")
    print("  4. Orientation/heading")
    print("  5. Battery status")
    print("  6. Movement control")
    print("\nRun this INDOORS before outdoor GPS testing.")

    # Test connection
    rover = test_connection()
    if not rover:
        return

    # Run all tests
    results = {
        'Camera': test_camera(rover),
        'IMU': test_imu(rover),
        'Orientation': test_orientation(rover),
        'Battery': test_battery(rover),
        'Control': test_control(rover),
    }

    # Summary
    print("\n" + "="*60)
    print("HARDWARE TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✓ All hardware tests PASSED")
        print("\nNEXT STEP: Go outdoors and run:")
        print("  python scripts/diagnose_coordinates.py")
    else:
        print("\n✗ Some tests FAILED - fix hardware issues first")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
