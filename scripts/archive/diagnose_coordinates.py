"""
Diagnostic script to check if coordinate transforms are correct
"""

import sys
import os
import numpy as np
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from earthrover_interface import EarthRoverInterface
from coordinate_utils import CoordinateTransformer, haversine_distance, gps_to_local, rotate_to_robot_frame

def main():
    parser = argparse.ArgumentParser(description="Coordinate Transform Diagnostic")
    parser.add_argument('--show-gps', action='store_true', help='Show current GPS position and exit')
    parser.add_argument('--target-lat', type=float, help='Target latitude')
    parser.add_argument('--target-lon', type=float, help='Target longitude')
    parser.add_argument('--distance', type=float, help='Estimated distance in meters')
    args = parser.parse_args()

    rover = EarthRoverInterface(timeout=30.0)
    if not rover.connect():
        print("Failed to connect")
        return

    # Get current position
    lat, lon, heading_rad = rover.get_pose()
    if lat is None or lon is None:
        print("No GPS data")
        return

    heading_deg = np.degrees(heading_rad) if heading_rad else 0

    # Show GPS mode
    if args.show_gps:
        print("="*60)
        print("CURRENT GPS POSITION")
        print("="*60)
        print(f"\nLatitude:  {lat}")
        print(f"Longitude: {lon}")
        print(f"Heading:   {heading_deg:.1f}°")
        print(f"\nGoogle Maps: https://maps.google.com/?q={lat},{lon}")
        print(f"\nRecord this position, then:")
        print(f"1. Move robot forward or pick a landmark ahead")
        print(f"2. Run --show-gps again to get second position")
        print(f"3. Test with: python diagnose_coordinates.py --target-lat LAT --target-lon LON")
        return

    print("="*60)
    print("COORDINATE DIAGNOSTIC")
    print("="*60)

    print(f"\nCurrent Position:")
    print(f"  Latitude:  {lat}")
    print(f"  Longitude: {lon}")
    print(f"  Heading:   {heading_deg:.1f}° (0=North, 90=East, 180=South, 270=West)")
    print(f"  Heading (rad): {heading_rad:.3f}")

    # Get target coordinates
    if args.target_lat is None or args.target_lon is None:
        print(f"\nUsage:")
        print(f"  1. Show current GPS:  python diagnose_coordinates.py --show-gps")
        print(f"  2. Test coordinates:  python diagnose_coordinates.py --target-lat LAT --target-lon LON")
        return

    target_lat = args.target_lat
    target_lon = args.target_lon
    estimated_dist = args.distance if args.distance is not None else 0.0

    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    # Calculate distance
    actual_dist = haversine_distance(lat, lon, target_lat, target_lon)
    print(f"\nActual distance: {actual_dist:.1f}m")
    print(f"You estimated:   {estimated_dist:.1f}m")
    dist_error = abs(actual_dist - estimated_dist)
    if dist_error < 2:
        print("✓ Distance matches!")
    else:
        print(f"✗ Distance off by {dist_error:.1f}m - GPS might be inaccurate")

    # Convert to local frame
    x_world, y_world = gps_to_local(lat, lon, target_lat, target_lon)
    print(f"\nWorld frame (X=East, Y=North):")
    print(f"  Target is at: ({x_world:.1f}, {y_world:.1f})m")

    # Convert to robot frame
    x_robot, y_robot = rotate_to_robot_frame(x_world, y_world, heading_rad)
    print(f"\nRobot frame (X=Forward, Y=Left):")
    print(f"  Target is at: ({x_robot:.1f}, {y_robot:.1f})m")

    # Interpret
    print(f"\nInterpretation:")
    if x_robot > 1:
        print(f"  ✓ Target is AHEAD ({x_robot:.1f}m forward)")
    elif x_robot < -1:
        print(f"  ✗ Target is BEHIND ({-x_robot:.1f}m backward) - THIS IS WRONG!")
    else:
        print(f"  ~ Target is roughly at robot's position")

    if y_robot > 1:
        print(f"  ? Target is to the LEFT ({y_robot:.1f}m)")
    elif y_robot < -1:
        print(f"  ? Target is to the RIGHT ({-y_robot:.1f}m)")
    else:
        print(f"  ✓ Target is roughly centered")

    print(f"\nExpected: Target should be AHEAD (X > 0) and CENTERED (Y ≈ 0)")
    print(f"If X is negative or Y is large, coordinates are WRONG!")

    # Check with CoordinateTransformer
    print(f"\n{'='*60}")
    print("TRANSFORMER TEST")
    print(f"{'='*60}")

    transformer = CoordinateTransformer(history_size=5)

    # Feed current position 5 times
    for i in range(5):
        transformer.update(lat, lon, heading_rad)

    coords = transformer.get_model_input(target_lat, target_lon)
    target_from_transformer = coords[5]

    print(f"\nCoordinateTransformer output:")
    print(f"  Target: ({target_from_transformer[0]:.1f}, {target_from_transformer[1]:.1f})m")

    if abs(target_from_transformer[0] - x_robot) < 0.5 and abs(target_from_transformer[1] - y_robot) < 0.5:
        print("  ✓ Transformer matches manual calculation!")
    else:
        print("  ✗ Transformer gives DIFFERENT result - BUG IN TRANSFORMER!")

if __name__ == "__main__":
    main()
