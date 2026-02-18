# What Went Wrong & Lessons Learned

## Overview
Deploying CityWalker (CVPR 2025) on the FrodoBots EarthRover. Multiple mistakes
were made because we didn't properly read the CityWalker source code before writing
deployment code.

---

## Mistake 1: Used a Fictional SDK
**What we did:** Imported `earth_rovers_sdk.get_earth_rover_service` — a module that
doesn't exist.

**Why:** Assumed the SDK had a high-level Python package. In reality, the EarthRover
SDK is a FastAPI server (`earth-rovers-sdk/`) running on localhost:8000, and the
actual Python client is `src/earthrover_interface.py` → `EarthRoverInterface`.

**Fix:** Use `EarthRoverInterface` with methods: `get_camera_frame()`, `get_pose()`,
`send_control(linear, angular)`, `stop()`.

---

## Mistake 2: Wrong step_scale (0.3 instead of ~1.0-1.5)
**What we did:** Used `step_scale = 0.3` (or `robot_speed / inference_rate`), a made-up
value.

**Why:** Didn't read the training data code. The actual `step_scale` is the average
distance between consecutive GPS positions at 1Hz, computed in
`CityWalker/data/teleop_dataset.py`:
```python
step_scale = np.linalg.norm(np.diff(pose[:, [0, 1]], axis=0), axis=1).mean()
```
For walking speed (~1.0-1.5 m/s at 1Hz), step_scale ~ 1.0-1.5m.

**Fix:** Compute step_scale dynamically from actual GPS trajectory history.

---

## Mistake 3: Wrong Coordinate Transform (Compass vs GPS Trajectory)
**What we did:** Used compass heading to rotate coordinates into robot frame.

**Why:** Assumed heading = compass bearing. The actual CityWalker approach (from
`teleop_dataset.py` → `transform_input()`) derives heading from the GPS trajectory
direction:
```python
# Translate so current position = origin
translated = positions - current_position
# Derive heading from second-to-last position direction
second_last = translated[-2]
angle = -np.pi/2 - np.arctan2(second_last[1], second_last[0])
# Rotate all positions by this angle
rotated = translated @ rotation_matrix.T
```
This makes +Y = forward movement direction, matching training data.

**Fix:** Implement exact `transform_input()` from teleop_dataset.py.

---

## Mistake 4: Control Values in Wrong Units
**What we did:** PD controller output in m/s.

**Why:** Didn't check `send_control()` interface. It takes values clamped to [-1, 1]
(throttle-style), not velocities.

**Fix:** Clip PD output to [-1, 1] before sending.

---

## Mistake 5: Wrong Inference Rate (2Hz instead of 1Hz)
**What we did:** Default inference at 2Hz.

**Why:** Didn't check training config. `target_fps=1` in
`CityWalker/config/citywalk_2000hr.yaml` means 1 frame per second, and each predicted
waypoint = 1 second into the future.

**Fix:** Run inference at 1Hz.

---

## Mistake 6: No GPS History Buffer
**What we did:** Stored dummy zero-filled coordinate buffer.

**Why:** CityWalker needs 5 real past GPS positions (in local meters) to understand
movement context. We just fed zeros, so the model had no trajectory history.

**Fix:** Maintain a deque of the last 5 GPS positions in local ENU meters, converted
via `latlon_to_local()`.

---

## Mistake 7: Video Tests with Fake Coordinates
**What we did:** Tested CityWalker on sample street videos with simulated straight-line
coordinates (all zeros for x, incrementing y).

**Why:** Thought we could validate the model without real GPS data.

**Result:** Waypoints all pointed straight ahead regardless of video content — completely
meaningless output. CityWalker needs real GPS trajectories to make meaningful predictions.

---

## Mistake 8: Spiraling During Early Robot Tests
**What we did:** Robot spiraled in circles instead of navigating.

**Why:** Combination of wrong step_scale (0.3), wrong coordinate transform, and no
actual GPS history feeding into the model. The waypoints were garbage, so the PD
controller drove in circles.

---

## Root Cause
**We didn't read the CityWalker source code properly.** Everything we needed was in:
- `CityWalker/data/teleop_dataset.py` — GPS handling, coordinate transform, step_scale
- `CityWalker/model/citywalker_feat.py` — Image preprocessing, model forward pass
- `CityWalker/config/citywalk_2000hr.yaml` — Training config (target_fps, crop sizes)
- `CityWalker/pl_modules/citywalker_feat_module.py` — Training loop, denormalization

The deployment code was never provided in the CityWalker repo (legitimate gap), but
ALL the coordinate handling and preprocessing details were clearly in the source.

---

## What's Correct Now
- `src/citywalker_wrapper.py` — Clean model interface with proper step_scale handling
- `scripts/navigate_citywalker.py` — Correct GPS transform, 1Hz inference, [-1,1] control
- Next step: Integrate DBR (Depth Barrier Regularization) for obstacle avoidance
