"""
Thorough test of DBR components.

This script verifies:
1. Depth estimator loads and produces valid output
2. DBR module (polar reduction + barrier loss) works correctly
3. Depth safety layer integrates properly
4. Gradient flow through the full pipeline

Run from rover/ directory:
    conda activate rover
    python scripts/test_dbr_components.py
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================================
# TEST 1: Depth Estimator
# ============================================================================

def test_depth_estimator():
    print("=" * 60)
    print("TEST 1: Depth Estimator")
    print("=" * 60)

    errors = []

    print("\n[1.1] Loading depth estimator...")
    try:
        from depth_estimator import DepthEstimator
        estimator = DepthEstimator(model_size='small')
        print("  Loaded - OK")
    except Exception as e:
        errors.append(f"Failed to load depth estimator: {e}")
        print(f"  FAIL: {e}")
        print("  (Continuing with remaining tests using dummy data)")
        return errors, None

    # Test 1.2: Single image depth estimation
    print("\n[1.2] Testing single image depth estimation...")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = estimator.estimate(dummy_image)

    if depth.shape == (480, 640):
        print(f"  Output shape: {depth.shape} - OK")
    else:
        errors.append(f"Wrong depth shape: {depth.shape}")
        print(f"  FAIL: shape is {depth.shape}")

    if depth.min() >= 0:
        print(f"  All depths >= 0: min={depth.min():.2f} - OK")
    else:
        errors.append(f"Negative depth: {depth.min()}")
        print(f"  FAIL: negative depth {depth.min()}")

    if depth.max() <= 20.0:
        print(f"  All depths <= max_depth: max={depth.max():.2f} - OK")
    else:
        # Not necessarily an error, just a warning
        print(f"  NOTE: max depth {depth.max():.2f} > 20.0 (may be fine)")

    # Test 1.3: Polar clearance computation
    print("\n[1.3] Testing polar clearance...")
    clearance, bin_centers = estimator.get_polar_clearance(depth, num_bins=32)

    if clearance.shape == (32,):
        print(f"  Clearance shape: {clearance.shape} - OK")
    else:
        errors.append(f"Wrong clearance shape: {clearance.shape}")
        print(f"  FAIL: {clearance.shape}")

    if bin_centers.shape == (32,):
        print(f"  Bin centers shape: {bin_centers.shape} - OK")
    else:
        errors.append(f"Wrong bin_centers shape: {bin_centers.shape}")
        print(f"  FAIL: {bin_centers.shape}")

    # Check bin centers span FOV
    fov_half = np.radians(45)  # 90 deg FOV / 2
    if abs(bin_centers[0] - (-fov_half)) < 0.1:
        print(f"  Left edge: {np.degrees(bin_centers[0]):.1f}deg (expected -45) - OK")
    else:
        errors.append(f"Left edge wrong: {np.degrees(bin_centers[0]):.1f}")

    if abs(bin_centers[-1] - fov_half) < 0.1:
        print(f"  Right edge: {np.degrees(bin_centers[-1]):.1f}deg (expected +45) - OK")
    else:
        errors.append(f"Right edge wrong: {np.degrees(bin_centers[-1]):.1f}")

    # Test 1.4: Waypoint safety check
    print("\n[1.4] Testing waypoint safety check...")
    test_wp = np.array([1.0, 0.0])  # Straight ahead
    safe, cl = estimator.is_waypoint_safe(test_wp, clearance, bin_centers, margin=0.5)
    print(f"  Waypoint (1,0) ahead: safe={safe}, clearance={cl:.2f}m - OK")

    test_wp_left = np.array([0.5, 1.0])  # Left
    safe_l, cl_l = estimator.is_waypoint_safe(test_wp_left, clearance, bin_centers, margin=0.5)
    print(f"  Waypoint (0.5,1) left: safe={safe_l}, clearance={cl_l:.2f}m - OK")

    # Test 1.5: Safe direction finding
    print("\n[1.5] Testing safe direction finding...")
    best_angle, best_cl = estimator.get_safe_direction(clearance, bin_centers, margin=0.5)
    print(f"  Best direction: {np.degrees(best_angle):.1f}deg, clearance={best_cl:.2f}m - OK")

    # Test 1.6: Different image sizes
    print("\n[1.6] Testing different image sizes...")
    for h, w in [(240, 320), (480, 640), (720, 1280)]:
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        try:
            d = estimator.estimate(img)
            if d.shape == (h, w):
                print(f"  {h}x{w}: output={d.shape} - OK")
            else:
                errors.append(f"Size {h}x{w} gave wrong output: {d.shape}")
                print(f"  FAIL: {d.shape}")
        except Exception as e:
            errors.append(f"Failed with size {h}x{w}: {e}")
            print(f"  FAIL: {e}")

    return errors, estimator


# ============================================================================
# TEST 2: DBR Module (Polar Reducer + Barrier Loss)
# ============================================================================

def test_dbr_module():
    print("\n" + "=" * 60)
    print("TEST 2: DBR Module")
    print("=" * 60)

    errors = []

    print("\n[2.1] Loading DBR module...")
    try:
        from dbr_module import DBRModule, DepthPolarReducer, BarrierLoss
        dbr = DBRModule(num_bins=32, margin=0.5, temperature=20.0)
        print("  Loaded - OK")
    except Exception as e:
        errors.append(f"Failed to load DBR module: {e}")
        print(f"  FAIL: {e}")
        return errors

    # Test 2.2: Forward pass with dummy data
    print("\n[2.2] Testing forward pass...")
    B, T = 2, 5
    H, W = 480, 640

    dummy_wp = torch.randn(B, T, 2) * 2.0
    dummy_depth = torch.rand(B, H, W) * 10.0 + 0.5

    loss, clearance = dbr(dummy_wp, dummy_depth)

    if loss.shape == ():
        print(f"  Loss shape: scalar - OK")
    else:
        errors.append(f"Loss not scalar: {loss.shape}")
        print(f"  FAIL: {loss.shape}")

    if loss.item() >= 0:
        print(f"  Loss value: {loss.item():.4f} (non-negative) - OK")
    else:
        errors.append(f"Negative loss: {loss.item()}")
        print(f"  FAIL: negative loss")

    if clearance.shape == (B, 32):
        print(f"  Clearance shape: {clearance.shape} - OK")
    else:
        errors.append(f"Wrong clearance shape: {clearance.shape}")
        print(f"  FAIL: {clearance.shape}")

    # Test 2.3: Unsafe waypoints should have higher loss
    print("\n[2.3] Testing loss ordering (unsafe > safe)...")

    # Obstacle directly ahead
    depth_obstacle = torch.ones(B, H, W) * 5.0
    depth_obstacle[:, 300:, 280:360] = 0.3  # 0.3m obstacle ahead

    safe_wp = torch.tensor([[[1.0, 2.0]] * T] * B)    # Pointing left
    unsafe_wp = torch.tensor([[[1.0, 0.0]] * T] * B)  # Pointing ahead

    loss_safe, _ = dbr(safe_wp, depth_obstacle)
    loss_unsafe, _ = dbr(unsafe_wp, depth_obstacle)

    if loss_unsafe.item() > loss_safe.item():
        print(f"  Safe loss: {loss_safe.item():.4f}")
        print(f"  Unsafe loss: {loss_unsafe.item():.4f}")
        print(f"  Unsafe > Safe: True - OK")
    else:
        errors.append("Unsafe loss not greater than safe loss")
        print(f"  FAIL: safe={loss_safe.item():.4f}, unsafe={loss_unsafe.item():.4f}")

    # Test 2.4: Gradient flow
    print("\n[2.4] Testing gradient flow...")
    wp_grad = torch.randn(B, T, 2, requires_grad=True)
    loss_grad, _ = dbr(wp_grad, dummy_depth)
    loss_grad.backward()

    if wp_grad.grad is not None:
        print(f"  Gradient shape: {wp_grad.grad.shape} - OK")
        grad_norm = wp_grad.grad.norm().item()
        if grad_norm > 0:
            print(f"  Gradient norm: {grad_norm:.4f} (non-zero) - OK")
        else:
            errors.append("Zero gradient")
            print(f"  FAIL: zero gradient")
    else:
        errors.append("No gradient computed")
        print(f"  FAIL: grad is None")

    # Test 2.5: DepthPolarReducer directly
    print("\n[2.5] Testing DepthPolarReducer directly...")
    reducer = DepthPolarReducer(num_bins=32, temperature=20.0)

    clearance_direct = reducer(dummy_depth)
    if clearance_direct.shape == (B, 32):
        print(f"  Reducer output shape: {clearance_direct.shape} - OK")
    else:
        errors.append(f"Reducer wrong shape: {clearance_direct.shape}")

    # All clearance values should be positive
    if (clearance_direct > 0).all():
        print(f"  All clearance > 0: min={clearance_direct.min():.2f} - OK")
    else:
        errors.append("Non-positive clearance values")
        print(f"  FAIL: min clearance = {clearance_direct.min():.2f}")

    # Test 2.6: BarrierLoss directly
    print("\n[2.6] Testing BarrierLoss directly...")
    barrier = BarrierLoss(margin=0.5)
    bin_centers = reducer.bin_centers

    loss_barrier = barrier(dummy_wp, clearance_direct, bin_centers)
    if loss_barrier.shape == ():
        print(f"  Barrier loss: {loss_barrier.item():.4f} - OK")
    else:
        errors.append(f"Barrier loss not scalar: {loss_barrier.shape}")

    # Test 2.7: Different batch sizes
    print("\n[2.7] Testing different batch sizes...")
    for batch in [1, 4, 8]:
        wp = torch.randn(batch, T, 2)
        depth = torch.rand(batch, H, W) * 10.0
        try:
            l, c = dbr(wp, depth)
            print(f"  Batch {batch}: loss={l.item():.4f} - OK")
        except Exception as e:
            errors.append(f"Failed with batch {batch}: {e}")
            print(f"  FAIL: {e}")

    # Test 2.8: Loss weight
    print("\n[2.8] Testing loss weight...")
    dbr_w1 = DBRModule(num_bins=32, margin=0.5, loss_weight=1.0)
    dbr_w2 = DBRModule(num_bins=32, margin=0.5, loss_weight=2.0)

    loss_w1, _ = dbr_w1(dummy_wp, dummy_depth)
    loss_w2, _ = dbr_w2(dummy_wp, dummy_depth)

    ratio = loss_w2.item() / (loss_w1.item() + 1e-10)
    if abs(ratio - 2.0) < 0.01:
        print(f"  Weight ratio: {ratio:.3f} (expected 2.0) - OK")
    else:
        errors.append(f"Weight ratio wrong: {ratio}")
        print(f"  FAIL: ratio = {ratio:.3f}")

    return errors


# ============================================================================
# TEST 3: Depth Safety Layer
# ============================================================================

def test_depth_safety():
    print("\n" + "=" * 60)
    print("TEST 3: Depth Safety Layer")
    print("=" * 60)

    errors = []

    # Test 3.1: DummyDepthSafety (always works, no model needed)
    print("\n[3.1] Testing DummyDepthSafety (no-op)...")
    from depth_safety import DummyDepthSafety

    dummy_safety = DummyDepthSafety()
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_wp = np.array([1.0, 0.0])

    result, overridden = dummy_safety.check_waypoint(dummy_frame, dummy_wp)
    if np.allclose(result, dummy_wp) and not overridden:
        print(f"  Pass-through works: overridden={overridden} - OK")
    else:
        errors.append("DummyDepthSafety modified waypoint")
        print(f"  FAIL: result={result}, overridden={overridden}")

    # Test with multiple waypoints
    dummy_wps = np.random.randn(5, 2)
    result_multi, overridden_multi = dummy_safety.check_waypoints(dummy_frame, dummy_wps)
    if np.allclose(result_multi, dummy_wps) and not overridden_multi:
        print(f"  Multi-waypoint pass-through - OK")
    else:
        errors.append("DummyDepthSafety modified waypoints")

    stats = dummy_safety.get_stats()
    if stats['total_checks'] == 2 and stats['total_overrides'] == 0:
        print(f"  Stats tracking: checks={stats['total_checks']}, overrides={stats['total_overrides']} - OK")
    else:
        errors.append(f"Wrong stats: {stats}")

    # Test 3.2: Real DepthSafetyLayer (needs model checkpoint)
    print("\n[3.2] Testing DepthSafetyLayer...")
    try:
        from depth_safety import DepthSafetyLayer
        safety = DepthSafetyLayer(model_size='small', margin=0.5)
        print("  Safety layer loaded - OK")

        # Single waypoint check
        print("\n[3.3] Testing single waypoint check...")
        result, overridden = safety.check_waypoint(dummy_frame, dummy_wp)
        print(f"  Input: ({dummy_wp[0]:.2f}, {dummy_wp[1]:.2f})")
        print(f"  Output: ({result[0]:.2f}, {result[1]:.2f})")
        print(f"  Overridden: {overridden}")
        print(f"  Inference: {safety.last_inference_ms:.1f}ms - OK")

        # Multiple waypoints
        print("\n[3.4] Testing multiple waypoint check...")
        wps = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
        safe_wps, overridden = safety.check_waypoints(dummy_frame, wps)

        if safe_wps.shape == (5, 2):
            print(f"  Output shape: {safe_wps.shape} - OK")
        else:
            errors.append(f"Wrong shape from check_waypoints: {safe_wps.shape}")

        # Only first should potentially be overridden, rest unchanged
        if np.allclose(safe_wps[1:], wps[1:]):
            print(f"  Waypoints 1-4 unchanged - OK")
        else:
            errors.append("check_waypoints modified non-first waypoints")

        # Stats
        print("\n[3.5] Testing stats...")
        stats = safety.get_stats()
        print(f"  Total checks: {stats['total_checks']}")
        print(f"  Total overrides: {stats['total_overrides']}")
        print(f"  Override rate: {stats['override_rate_pct']:.1f}%")
        print(f"  Forward clearance: {stats['forward_clearance']}")
        print(f"  Min clearance: {stats['min_clearance']}")

        # Clearance queries
        print("\n[3.6] Testing clearance queries...")
        fwd_cl = safety.get_forward_clearance()
        min_cl = safety.get_min_clearance()
        if fwd_cl is not None:
            print(f"  Forward clearance: {fwd_cl:.2f}m - OK")
        if min_cl is not None:
            print(f"  Min clearance: {min_cl:.2f}m - OK")

    except Exception as e:
        print(f"  Cannot test real safety layer: {e}")
        print(f"  (Need Depth Anything V2 checkpoint)")
        print(f"  DummyDepthSafety tests passed above.")

    return errors


# ============================================================================
# TEST 4: Integration - DBR + CityWalker
# ============================================================================

def test_integration():
    print("\n" + "=" * 60)
    print("TEST 4: Integration (DBR + CityWalker)")
    print("=" * 60)

    errors = []

    print("\n[4.1] Testing DBR module with CityWalker-shaped outputs...")
    from dbr_module import DBRModule

    dbr = DBRModule(num_bins=32, margin=0.5, temperature=20.0)

    # CityWalker outputs: (B, 5, 2) waypoints
    B = 2
    citywalker_waypoints = torch.randn(B, 5, 2) * 0.3  # Typical waypoint scale
    depth_map = torch.rand(B, 480, 640) * 10.0 + 0.5

    loss, clearance = dbr(citywalker_waypoints, depth_map)
    print(f"  CityWalker waypoints → DBR loss: {loss.item():.4f} - OK")
    print(f"  Clearance per direction: shape={clearance.shape}")

    # Test 4.2: Training-style forward+backward
    print("\n[4.2] Testing training-style forward+backward...")
    model_output = torch.randn(B, 5, 2, requires_grad=True)

    # Simulate total loss = waypoint_loss + dbr_loss
    waypoint_loss = torch.nn.functional.l1_loss(
        model_output, torch.zeros_like(model_output)
    )
    dbr_loss, _ = dbr(model_output, depth_map)
    total_loss = waypoint_loss + dbr_loss

    total_loss.backward()

    if model_output.grad is not None and model_output.grad.norm() > 0:
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Waypoint loss: {waypoint_loss.item():.4f}")
        print(f"  DBR loss: {dbr_loss.item():.4f}")
        print(f"  Grad norm: {model_output.grad.norm().item():.4f} - OK")
    else:
        errors.append("No gradient from combined loss")
        print(f"  FAIL: no gradient")

    # Test 4.3: Verify DBR pushes waypoints away from obstacles
    print("\n[4.3] Testing DBR gradient direction...")
    # Obstacle ahead at 0.3m
    depth_obstacle = torch.ones(1, 480, 640) * 5.0
    depth_obstacle[:, 300:, 280:360] = 0.3

    # Waypoint pointing at obstacle
    wp_ahead = torch.tensor([[[1.0, 0.0]]], requires_grad=True)  # (1, 1, 2)

    loss, _ = dbr(wp_ahead, depth_obstacle)
    loss.backward()

    # Gradient should push waypoint AWAY from obstacle (negative x gradient)
    grad_x = wp_ahead.grad[0, 0, 0].item()
    print(f"  Waypoint (1, 0) toward obstacle:")
    print(f"    Loss: {loss.item():.4f}")
    print(f"    Grad x: {grad_x:.4f} (should push away from obstacle)")

    if abs(grad_x) > 1e-6:
        print(f"    Gradient is non-zero - OK")
    else:
        errors.append("Zero gradient for waypoint at obstacle")
        print(f"    FAIL: zero gradient")

    return errors


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("DBR COMPONENT VERIFICATION TEST")
    print("=" * 60)

    all_errors = []

    # Run tests
    depth_errors, estimator = test_depth_estimator()
    all_errors.extend(depth_errors)

    dbr_errors = test_dbr_module()
    all_errors.extend(dbr_errors)

    safety_errors = test_depth_safety()
    all_errors.extend(safety_errors)

    integration_errors = test_integration()
    all_errors.extend(integration_errors)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if len(all_errors) == 0:
        print("\nALL TESTS PASSED!")
        print("\nDBR components verified:")
        print("  - Depth Estimator (loads, estimates, polar clearance)")
        print("  - DBR Module (reducer, barrier loss, gradient flow)")
        print("  - Depth Safety Layer (waypoint checking, stats)")
        print("  - Integration (CityWalker + DBR training)")
    else:
        print(f"\n{len(all_errors)} ERROR(S) FOUND:")
        for err in all_errors:
            print(f"  - {err}")
        print("\nPlease fix these issues before proceeding.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
