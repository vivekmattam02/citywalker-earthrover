"""
Thorough test of Step 1 (Model Wrapper) and Step 3 (Coordinate Utils).

This script verifies:
1. Model wrapper loads and runs correctly
2. Coordinate utils math is correct
3. Both components work together

Run from rover/ directory:
    conda activate rover
    python scripts/test_components.py
"""

import sys
import os
import numpy as np
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ============================================================================
# TEST 1: Coordinate Utils - Math Verification
# ============================================================================

def test_coordinate_utils():
    print("=" * 60)
    print("TEST 1: Coordinate Utilities")
    print("=" * 60)

    from src.coordinate_utils import (
        haversine_distance,
        gps_to_local,
        rotate_to_robot_frame,
        CoordinateTransformer
    )

    errors = []

    # Test 1.1: Haversine distance
    print("\n[1.1] Testing haversine distance...")

    # Known distance: NYC to LA is approximately 3,944 km
    nyc_lat, nyc_lon = 40.7128, -74.0060
    la_lat, la_lon = 34.0522, -118.2437
    dist = haversine_distance(nyc_lat, nyc_lon, la_lat, la_lon)
    expected = 3944000  # meters
    tolerance = 50000   # 50km tolerance

    if abs(dist - expected) < tolerance:
        print(f"  NYC to LA: {dist/1000:.0f} km (expected ~3944 km) - OK")
    else:
        errors.append(f"Haversine: got {dist/1000:.0f} km, expected ~3944 km")
        print(f"  FAIL: got {dist/1000:.0f} km")

    # Test 1.2: Small distance (should be ~111 meters for 0.001 degree latitude)
    print("\n[1.2] Testing small distances...")
    dist_small = haversine_distance(40.0, -74.0, 40.001, -74.0)
    expected_small = 111  # ~111 meters per 0.001 degree
    if abs(dist_small - expected_small) < 5:
        print(f"  0.001 degree lat: {dist_small:.1f} m (expected ~111 m) - OK")
    else:
        errors.append(f"Small distance: got {dist_small:.1f} m, expected ~111 m")
        print(f"  FAIL: got {dist_small:.1f} m")

    # Test 1.3: GPS to local conversion
    print("\n[1.3] Testing GPS to local conversion...")

    origin_lat, origin_lon = 40.0, -74.0

    # Point 100m north
    north_lat = origin_lat + 0.0009  # ~100m north
    x, y = gps_to_local(origin_lat, origin_lon, north_lat, origin_lon)
    if abs(x) < 1 and abs(y - 100) < 10:
        print(f"  Point north: x={x:.1f}m, y={y:.1f}m (expected x~0, y~100) - OK")
    else:
        errors.append(f"GPS north: x={x:.1f}, y={y:.1f}")
        print(f"  FAIL: x={x:.1f}, y={y:.1f}")

    # Point 100m east
    east_lon = origin_lon + 0.00117  # ~100m east at this latitude
    x, y = gps_to_local(origin_lat, origin_lon, origin_lat, east_lon)
    if abs(x - 100) < 15 and abs(y) < 1:
        print(f"  Point east: x={x:.1f}m, y={y:.1f}m (expected x~100, y~0) - OK")
    else:
        errors.append(f"GPS east: x={x:.1f}, y={y:.1f}")
        print(f"  FAIL: x={x:.1f}, y={y:.1f}")

    # Test 1.4: Robot frame rotation
    print("\n[1.4] Testing robot frame rotation...")

    # Robot facing north (heading=0), point is north
    # Should appear directly in front (positive x in robot frame)
    x_world, y_world = 0, 10  # 10m north in world frame
    heading = 0  # facing north
    x_rob, y_rob = rotate_to_robot_frame(x_world, y_world, heading)
    if abs(x_rob - 10) < 0.1 and abs(y_rob) < 0.1:
        print(f"  Facing north, point north: robot_x={x_rob:.1f}, robot_y={y_rob:.1f} - OK")
    else:
        errors.append(f"Rotation north: x={x_rob:.1f}, y={y_rob:.1f}")
        print(f"  FAIL: x={x_rob:.1f}, y={y_rob:.1f}")

    # Robot facing east (heading=pi/2), point is north
    # Point should appear to the left (positive y in robot frame)
    heading = math.pi / 2  # facing east
    x_rob, y_rob = rotate_to_robot_frame(x_world, y_world, heading)
    if abs(x_rob) < 0.1 and abs(y_rob - 10) < 0.1:
        print(f"  Facing east, point north: robot_x={x_rob:.1f}, robot_y={y_rob:.1f} - OK")
    else:
        errors.append(f"Rotation east: x={x_rob:.1f}, y={y_rob:.1f}")
        print(f"  FAIL: x={x_rob:.1f}, y={y_rob:.1f}")

    # Test 1.5: CoordinateTransformer
    print("\n[1.5] Testing CoordinateTransformer...")

    transformer = CoordinateTransformer(history_size=5)

    # Should not be ready before 5 updates
    if not transformer.is_ready():
        print("  Not ready before 5 updates - OK")
    else:
        errors.append("Transformer ready too early")
        print("  FAIL: ready too early")

    # Add 5 positions
    base_lat, base_lon = 40.0, -74.0
    for i in range(5):
        transformer.update(base_lat + i * 0.0001, base_lon, 0)

    if transformer.is_ready():
        print("  Ready after 5 updates - OK")
    else:
        errors.append("Transformer not ready after 5 updates")
        print("  FAIL: not ready after 5 updates")

    # Test output shape
    target_lat = base_lat + 0.001
    target_lon = base_lon
    coords = transformer.get_model_input(target_lat, target_lon)

    if coords.shape == (6, 2):
        print(f"  Output shape: {coords.shape} - OK")
    else:
        errors.append(f"Wrong shape: {coords.shape}")
        print(f"  FAIL: shape is {coords.shape}")

    if coords.dtype == np.float32:
        print(f"  Output dtype: {coords.dtype} - OK")
    else:
        errors.append(f"Wrong dtype: {coords.dtype}")
        print(f"  FAIL: dtype is {coords.dtype}")

    return errors


# ============================================================================
# TEST 2: Model Wrapper - Loading and Inference
# ============================================================================

def test_model_wrapper():
    print("\n" + "=" * 60)
    print("TEST 2: Model Wrapper")
    print("=" * 60)

    errors = []

    print("\n[2.1] Loading model wrapper...")
    try:
        from src.citywalker_wrapper import CityWalkerWrapper
        wrapper = CityWalkerWrapper()
        print("  Model loaded - OK")
    except Exception as e:
        errors.append(f"Failed to load model: {e}")
        print(f"  FAIL: {e}")
        return errors

    # Test 2.2: Check config values
    print("\n[2.2] Checking config values...")
    if wrapper.get_context_size() == 5:
        print(f"  Context size: {wrapper.get_context_size()} - OK")
    else:
        errors.append(f"Wrong context size: {wrapper.get_context_size()}")

    if wrapper.get_prediction_steps() == 5:
        print(f"  Prediction steps: {wrapper.get_prediction_steps()} - OK")
    else:
        errors.append(f"Wrong prediction steps: {wrapper.get_prediction_steps()}")

    # Test 2.3: Test with numpy input
    print("\n[2.3] Testing with numpy input...")
    dummy_images = np.random.randint(0, 255, (5, 480, 640, 3), dtype=np.uint8)
    dummy_coords = np.random.randn(6, 2).astype(np.float32)

    waypoints, arrived = wrapper.predict(dummy_images, dummy_coords)

    if waypoints.shape == (5, 2):
        print(f"  Waypoints shape: {waypoints.shape} - OK")
    else:
        errors.append(f"Wrong waypoints shape: {waypoints.shape}")
        print(f"  FAIL: {waypoints.shape}")

    if isinstance(arrived, float) and 0 <= arrived <= 1:
        print(f"  Arrival probability: {arrived:.4f} (valid range) - OK")
    else:
        errors.append(f"Invalid arrival value: {arrived}")
        print(f"  FAIL: {arrived}")

    # Test 2.4: Test with different image sizes
    print("\n[2.4] Testing with different image sizes...")
    for h, w in [(480, 640), (720, 1280), (240, 320)]:
        dummy_images = np.random.randint(0, 255, (5, h, w, 3), dtype=np.uint8)
        try:
            waypoints, arrived = wrapper.predict(dummy_images, dummy_coords)
            print(f"  Image size {h}x{w}: OK")
        except Exception as e:
            errors.append(f"Failed with image size {h}x{w}: {e}")
            print(f"  FAIL: {h}x{w} - {e}")

    # Test 2.5: Test step_length scaling
    print("\n[2.5] Testing step_length scaling...")
    waypoints_1, _ = wrapper.predict(dummy_images, dummy_coords, step_length=1.0)
    waypoints_2, _ = wrapper.predict(dummy_images, dummy_coords, step_length=2.0)

    # waypoints_2 should be exactly 2x waypoints_1
    ratio = waypoints_2 / (waypoints_1 + 1e-10)  # avoid division by zero
    if np.allclose(ratio, 2.0, atol=0.01):
        print(f"  Step length scaling: OK")
    else:
        errors.append("Step length scaling incorrect")
        print(f"  FAIL: scaling not working correctly")

    return errors


# ============================================================================
# TEST 3: Integration - Both Components Together
# ============================================================================

def test_integration():
    print("\n" + "=" * 60)
    print("TEST 3: Integration Test")
    print("=" * 60)

    errors = []

    print("\n[3.1] Loading both components...")
    try:
        from src.citywalker_wrapper import CityWalkerWrapper
        from src.coordinate_utils import CoordinateTransformer

        wrapper = CityWalkerWrapper()
        transformer = CoordinateTransformer(history_size=5)
        print("  Both components loaded - OK")
    except Exception as e:
        errors.append(f"Failed to load components: {e}")
        print(f"  FAIL: {e}")
        return errors

    # Simulate a robot scenario
    print("\n[3.2] Simulating robot movement...")

    # Robot starts at Times Square, moves north
    start_lat, start_lon = 40.7580, -73.9855
    target_lat, target_lon = 40.7590, -73.9855  # ~110m north

    # Simulate 5 position updates
    for i in range(5):
        lat = start_lat + i * 0.00002  # Small movements north
        transformer.update(lat, start_lon, heading=0)

    if transformer.is_ready():
        print("  Trajectory ready - OK")
    else:
        errors.append("Trajectory not ready")
        print("  FAIL")
        return errors

    # Get coordinates for model
    coords = transformer.get_model_input(target_lat, target_lon)
    print(f"  Coordinate shape: {coords.shape} - OK")

    # Create dummy images (in real scenario, these come from camera)
    dummy_images = np.random.randint(0, 255, (5, 480, 640, 3), dtype=np.uint8)

    # Run prediction
    print("\n[3.3] Running full prediction pipeline...")
    waypoints, arrived = wrapper.predict(dummy_images, coords, step_length=0.1)

    print(f"  Waypoints (meters):")
    for i, (x, y) in enumerate(waypoints):
        print(f"    Step {i+1}: x={x:.3f}m, y={y:.3f}m")
    print(f"  Arrival probability: {arrived:.4f}")

    # Verify outputs are reasonable
    if waypoints.shape == (5, 2):
        print("\n  Output shape correct - OK")
    else:
        errors.append(f"Wrong output shape: {waypoints.shape}")

    # Check distance to target
    dist = transformer.get_distance_to_target(target_lat, target_lon)
    print(f"  Distance to target: {dist:.1f}m")

    return errors


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("COMPONENT VERIFICATION TEST")
    print("=" * 60)

    all_errors = []

    # Run all tests
    all_errors.extend(test_coordinate_utils())
    all_errors.extend(test_model_wrapper())
    all_errors.extend(test_integration())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if len(all_errors) == 0:
        print("\nALL TESTS PASSED!")
        print("\nStep 1 (Model Wrapper) and Step 3 (Coordinate Utils) are verified.")
        print("Safe to proceed with next steps.")
    else:
        print(f"\n{len(all_errors)} ERROR(S) FOUND:")
        for err in all_errors:
            print(f"  - {err}")
        print("\nPlease fix these issues before proceeding.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
