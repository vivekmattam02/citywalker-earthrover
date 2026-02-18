"""
Check what IMU data is actually available from the SDK
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from earthrover_interface import EarthRoverInterface

def main():
    print("="*70)
    print("CHECKING IMU DATA AVAILABILITY")
    print("="*70)

    rover = EarthRoverInterface(timeout=30.0)
    if not rover.connect():
        print("✗ Failed to connect")
        return

    print("\n✓ Connected\n")
    print("Fetching IMU data for 5 seconds...\n")

    for i in range(5):
        print(f"\n{'='*70}")
        print(f"Sample {i+1}/5")
        print(f"{'='*70}")

        # Get raw data from SDK
        data = rover.get_data(use_cache=False)

        if data is None:
            print("✗ No data returned from SDK")
            time.sleep(1)
            continue

        # Check what keys are available
        print("\nAvailable data keys:")
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
                if len(value) > 0:
                    print(f"    First item: {value[0]}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")

        # Check IMU specifically
        print("\n" + "-"*70)
        imu_data = rover.get_imu()
        if imu_data:
            print("IMU data keys:")
            for key in sorted(imu_data.keys()):
                value = imu_data[key]
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} samples")
                    if len(value) > 0 and len(value) <= 3:
                        print(f"    Data: {value}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("✗ No IMU data")

        time.sleep(1)

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Final check
    data = rover.get_data(use_cache=False)
    if data:
        has_rpms = 'rpms' in data or ('imu' in data and isinstance(data.get('imu'), dict) and 'rpms' in data['imu'])
        has_gyros = 'gyroscope' in data or ('imu' in data and isinstance(data.get('imu'), dict) and 'gyroscope' in data['imu'])

        print(f"\n✓ SDK provides data: {sorted(data.keys())}")
        print(f"\nRPM data available: {'YES' if has_rpms else 'NO'}")
        print(f"Gyro data available: {'YES' if has_gyros else 'NO'}")

        if not has_rpms:
            print("\n⚠ RPM/wheel encoder data is NOT available from this robot!")
            print("   This means wheel odometry cannot be used.")
            print("   We need to use GPS + heading only (no wheel encoders).")


if __name__ == "__main__":
    main()
