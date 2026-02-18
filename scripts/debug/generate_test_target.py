"""
Generate GPS Target Coordinates

Creates a GPS target at a specified distance and direction from current position.
Useful for testing when you can't get accurate coordinates from maps.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from earthrover_interface import EarthRoverInterface


def main():
    print("="*60)
    print("GENERATE TEST TARGET COORDINATES")
    print("="*60)

    # Connect to robot
    rover = EarthRoverInterface(timeout=30.0)
    if not rover.connect():
        print("✗ Failed to connect to robot")
        return

    # Get current position
    lat, lon, heading_rad = rover.get_pose()
    if lat is None or lon is None:
        print("✗ No GPS data")
        return

    heading_deg = np.degrees(heading_rad) if heading_rad else 0

    print(f"\nCurrent Position:")
    print(f"  Latitude:  {lat:.8f}")
    print(f"  Longitude: {lon:.8f}")
    print(f"  Heading:   {heading_deg:.1f}°")
    print(f"\nGoogle Maps: https://maps.google.com/?q={lat},{lon}")

    # Get desired distance and direction
    print("\n" + "-"*60)
    distance_input = input("Distance to target (feet, default 15): ").strip()
    distance_ft = float(distance_input) if distance_input else 15.0
    distance_m = distance_ft * 0.3048

    print("\nDirection options:")
    print("  1. Straight ahead (robot's current heading)")
    print("  2. Behind robot")
    print("  3. Left of robot")
    print("  4. Right of robot")
    print("  5. Custom angle")

    direction = input("Choose direction (1-5, default 1): ").strip()

    if direction == "2":
        # Behind
        angle_offset = 180
    elif direction == "3":
        # Left
        angle_offset = 90
    elif direction == "4":
        # Right
        angle_offset = -90
    elif direction == "5":
        # Custom
        angle_offset = float(input("Angle offset from heading (degrees, + = left, - = right): "))
    else:
        # Straight ahead (default)
        angle_offset = 0

    # Calculate target heading
    target_heading_deg = (heading_deg + angle_offset) % 360
    target_heading_rad = np.radians(target_heading_deg)

    # Calculate offset in meters
    # East offset (x): distance * sin(heading)
    # North offset (y): distance * cos(heading)
    dx = distance_m * np.sin(target_heading_rad)
    dy = distance_m * np.cos(target_heading_rad)

    # Convert to GPS coordinates
    # 1 degree latitude ≈ 111139 meters
    # 1 degree longitude ≈ 111139 * cos(latitude) meters
    lat_change = dy / 111139.0
    lon_change = dx / (111139.0 * np.cos(np.radians(lat)))

    target_lat = lat + lat_change
    target_lon = lon + lon_change

    # Calculate actual distance for verification
    from coordinate_utils import haversine_distance
    actual_distance = haversine_distance(lat, lon, target_lat, target_lon)

    # Display results
    print("\n" + "="*60)
    print("TARGET COORDINATES")
    print("="*60)
    print(f"\nTarget Latitude:  {target_lat:.8f}")
    print(f"Target Longitude: {target_lon:.8f}")
    print(f"\nDistance: {distance_ft:.1f} feet ({distance_m:.1f} meters)")
    print(f"Direction: {target_heading_deg:.1f}° (heading)")
    print(f"Actual distance: {actual_distance:.2f}m")
    print(f"\nGoogle Maps: https://maps.google.com/?q={target_lat},{target_lon}")

    # Generate navigation command
    print("\n" + "="*60)
    print("NAVIGATION COMMAND")
    print("="*60)
    print("\nCopy and run this command:\n")
    print(f"python scripts/outdoor_nav_odometry.py \\")
    print(f"    --target-lat {target_lat:.8f} \\")
    print(f"    --target-lon {target_lon:.8f} \\")
    print(f"    --timeout 120")
    print()


if __name__ == "__main__":
    main()
