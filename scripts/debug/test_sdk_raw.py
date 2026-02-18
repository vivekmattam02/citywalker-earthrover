"""
Test SDK Raw Data

Directly query SDK endpoints to see if data is updating at all.
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def get_endpoint(endpoint):
    """Get data from SDK endpoint"""
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", timeout=5.0)
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error {response.status_code}"
    except Exception as e:
        return f"Exception: {e}"

def main():
    print("="*70)
    print("SDK RAW DATA TEST")
    print("="*70)
    print("\nQuerying SDK endpoints directly every 2 seconds...")
    print("If values don't change, SDK is frozen.\n")

    print("DRIVE THE ROBOT MANUALLY NOW!")
    print("(Open: http://localhost:8000/sdk)\n")

    time.sleep(3)

    for i in range(10):
        print(f"\n{'='*70}")
        print(f"READING #{i+1}")
        print(f"{'='*70}")

        # GPS
        gps_data = get_endpoint("/v2/gps")
        if isinstance(gps_data, dict):
            lat = gps_data.get('latitude')
            lon = gps_data.get('longitude')
            print(f"\nGPS (/v2/gps):")
            print(f"  Latitude:  {lat}")
            print(f"  Longitude: {lon}")
        else:
            print(f"\nGPS: {gps_data}")

        # IMU (includes orientation)
        imu_data = get_endpoint("/v2/imu")
        if isinstance(imu_data, dict):
            orientation = imu_data.get('orientation')
            gyros = imu_data.get('gyroscope', [])
            rpms = imu_data.get('rpms', [])
            print(f"\nIMU (/v2/imu):")
            print(f"  Orientation: {orientation}")
            print(f"  Gyro samples: {len(gyros) if isinstance(gyros, list) else 'N/A'}")
            print(f"  RPM samples: {len(rpms) if isinstance(rpms, list) else 'N/A'}")

            if isinstance(rpms, list) and len(rpms) > 0:
                print(f"  First RPM: {rpms[0]}")
            if isinstance(gyros, list) and len(gyros) > 0:
                print(f"  First Gyro: {gyros[0]}")
        else:
            print(f"\nIMU: {imu_data}")

        # Status
        status_data = get_endpoint("/v2/status")
        if isinstance(status_data, dict):
            battery = status_data.get('battery')
            print(f"\nStatus (/v2/status):")
            print(f"  Battery: {battery}%")
        else:
            print(f"\nStatus: {status_data}")

        # Control state
        control_data = get_endpoint("/v2/control")
        if isinstance(control_data, dict):
            print(f"\nControl (/v2/control):")
            print(f"  {json.dumps(control_data, indent=2)}")
        else:
            print(f"\nControl: {control_data}")

        print(f"\nWaiting 2 seconds...")
        time.sleep(2)

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("\nCheck the output above:")
    print("  - If GPS lat/lon are EXACTLY the same every time → GPS frozen")
    print("  - If Orientation is EXACTLY the same → Heading frozen")
    print("  - If Gyro/RPM samples are always 0 → No sensor data")
    print("\nIf all values are frozen, the SDK is not updating sensor data.")
    print("This could mean:")
    print("  1. Robot is not powered on / motors not enabled")
    print("  2. SDK server has crashed/frozen")
    print("  3. Robot's onboard computer is not running")
    print("  4. Communication lost between SDK and robot hardware")


if __name__ == "__main__":
    main()
