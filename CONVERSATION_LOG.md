# Conversation Log: CityWalker Deployment on EarthRover

## Session Overview
Two sessions working on deploying CityWalker (CVPR 2025) on FrodoBots EarthRover for outdoor GPS-based vision navigation.

---

## Session 1 (ran out of context)

### Phase 1: Initial Setup & Video Testing
- Had 2 sample street videos in `samples/` folder
- Created `scripts/test_video_citywalker.py` to run CityWalker on videos and visualize waypoints
- Fixed conda environment issues (`~/miniconda` not `~/miniconda3`)
- Successfully processed both videos, output saved to `samples/output/`
- Created `scripts/test_step_length_comparison.py` to compare 4 step_length values (0.1, 0.3, 0.6, 1.0)
- Comparison showed linear scaling — waypoints just scaled proportionally

### Phase 2: Realization Videos Were Useless
- User pointed out: "the images or the videos give nothing it is just random bs"
- Confirmed: waypoints all pointed straight ahead through people — meaningless
- Root cause: fake coordinates (all zeros for x, straight forward for y)
- CityWalker needs REAL GPS trajectories, not simulated straight-line motion

### Phase 3: User Called Out Lack of Source Code Reading
- User asked: "did you actually go through the paper and the other resources?"
- This was the turning point — I had NOT properly read the CityWalker source code
- All the answers were already in the repo

### Phase 4: Deep Source Code Analysis
Read the actual CityWalker source code thoroughly:

**`CityWalker/data/citywalk_feat_dataset.py`** — Training dataset
- Coordinate format, step_scale computation

**`CityWalker/model/citywalker_feat.py`** — Model architecture
- Forward pass: RGB normalize → center crop → resize → DINOv2 → coord embedding → transformer → cumsum waypoints
- Image preprocessing handled internally (ImageNet norm, crop to [350,630])

**`CityWalker/config/citywalk_2000hr.yaml`** — Training config
- `target_fps: 1` (1Hz inference)
- `context_size: 5` (5 past frames)
- `len_traj_pred: 5` (5 future waypoints)
- `crop: [350, 630]`, `resize: [350, 630]`

**`CityWalker/pl_modules/citywalker_feat_module.py`** — Training loop
- Output denormalization via `torch.cumsum()`

**`CityWalker/data/teleop_dataset.py`** — THE CRITICAL FILE (actual robot deployment code)
- `latlon_to_local(lat, lon, lat0, lon0)` using R_earth=6378137
- `transform_input()`: translate to current=origin, rotate so GPS trajectory direction = +Y
- Heading derived from GPS trajectory, NOT compass
- step_scale = `np.linalg.norm(np.diff(pose[:, [0, 1]], axis=0), axis=1).mean()`

**`CityWalker/config/finetune.yaml`** — Robot fine-tune config
- Different crop: [400,400] → [392,392] (square for robot camera)
- `data.type: teleop`

**`CityWalker/utils/gps_utils/app.py`** — Flask HTTPS server for phone GPS collection

### Phase 5: Identified All Problems

1. **Fictional SDK** — Imported `earth_rovers_sdk.get_earth_rover_service` which doesn't exist. Actual: `EarthRoverInterface`
2. **Wrong step_scale (0.3)** — Should be ~1.0-1.5m (average GPS step distance at 1Hz)
3. **Wrong coordinate transform** — Used compass heading. Should use GPS trajectory direction
4. **Control values in wrong units** — PD controller outputting m/s, but `send_control()` takes [-1, 1]
5. **Wrong inference rate (2Hz)** — Model trained at `target_fps=1`, should be 1Hz
6. **No GPS history buffer** — Fed zeros instead of real past positions
7. **Robot spiraling** — Result of all the above mistakes combined

### Phase 6: Wrote Plan & Rewrote Code

Created `PLAN.md` with detailed changes needed.

**Rewrote `src/citywalker_wrapper.py`:**
- Removed `robot_step_length` parameter
- Made `step_scale` required argument to `predict()`
- Clean documentation matching actual model behavior
- Key: images as numpy (5, H, W, 3) uint8, model handles preprocessing internally

**Rewrote `scripts/navigate_citywalker.py`:**
- Uses `EarthRoverInterface` (not fictional SDK)
- GPS history buffer (deque of 5 positions in local meters)
- `latlon_to_local()` matching teleop_dataset.py
- `transform_input()` matching teleop_dataset.py exactly:
  - Translate so current = origin
  - Derive heading from GPS trajectory (second-to-last → current direction)
  - Rotate so movement direction = +Y
- `compute_step_scale()` from actual GPS trajectory distances
- 1Hz inference (matching target_fps=1)
- Control values clipped to [-1, 1]
- Goal tolerance 5m

### Phase 7: Discovered DBR
- User pointed out `DBR/` directory
- Depth Barrier Regularization — safety extension to CityWalker
- Has FrodoBots-specific config: `config/frodobots_dbr.yaml`
- User said: "very important for the next step"

### Phase 8: User Requested Cleanup
User: "can you remove all the unnecessary ones but can you keep a copy of what is that we did wrong and why is that we did that in that way and keep a file for that and then delete all the unnecessary ones but keep the @DBR file that is very important for the next step"

---

## Session 2 (current)

### Cleanup Completed

**Created:**
- `WHAT_WENT_WRONG.md` — Documents all 8 mistakes and root causes

**Deleted (unnecessary docs — 13 files):**
- `CITYWALKER_DEPLOYMENT_COMPLETE.md` — Based on wrong assumptions
- `DEBUGGING_JOURNAL.md` — Old debugging notes
- `HOW_TO_RUN_CITYWALKER.md` — Wrong instructions (fictional SDK)
- `MEETING_GUIDE.txt` — Meeting is over
- `ODOMETRY_FUSION_GUIDE.md` — Blocked by no RPMs
- `OUTDOOR_GPS_NAVIGATION.md` — Old/wrong
- `PLAN.md` — Temporary planning file
- `START_HERE.md` — Pointed to wrong files
- `TODAYS_WORK_SUMMARY.md` — Described useless video tests
- `cursor_repository_purpose_explanation.md` — Old Cursor export
- `LEARNING.md` — Based on wrong understanding
- `PROGRESS.md` — Outdated progress tracker
- `TESTING_GUIDE.txt` — From earlier, outdated

**Deleted (wrong/unused scripts — 9 files):**
- `scripts/outdoor_nav_simple_gps.py` — Fictional SDK, wrong coords
- `scripts/calibrate_step_length.py` — Unnecessary
- `scripts/test_video_citywalker.py` — Useless (fake coordinates)
- `scripts/test_step_length_comparison.py` — Useless (fake coordinates)
- `scripts/outdoor_nav_odometry.py` — Wrong approach
- `scripts/run_nav2.py` — Abandoned ROS2 Nav2 attempt
- `scripts/setup_nav2.sh` — Abandoned
- `scripts/setup_nav2_conda.sh` — Abandoned
- `scripts/setup_nav2_simple.sh` — Abandoned

**Deleted (wrong/unused src modules — 8 files):**
- `src/coordinate_utils.py` — Old coordinate utils
- `src/dead_reckoning.py` — No wheel encoder data
- `src/direct_robot_interface.py` — Old interface
- `src/navigator.py` — Old orchestrator with wrong approach
- `src/odometry_fusion.py` — No RPMs available
- `src/orb_features.py` — Unused ORB feature tracking
- `src/pd_controller.py` — Not used by new navigate_citywalker.py
- `src/visual_odometry.py` — Unused

**Deleted (abandoned directories):**
- `ros2_ws/` — Entire ROS2 Nav2 workspace
- `samples/output/` — Output from useless video tests

### Final Clean Project Structure
```
rover/
├── CityWalker/              # CityWalker repo (reference code)
├── DBR/                     # Depth Barrier Regularization (NEXT STEP)
│   ├── config/
│   │   └── frodobots_dbr.yaml  # FrodoBots-specific config
│   ├── model/
│   ├── data/
│   └── ...
├── earth-rovers-sdk/        # EarthRover FastAPI SDK server
├── models/
│   └── CityWalker_2000hr.ckpt  # Pretrained model (1.7GB)
├── src/
│   ├── citywalker_wrapper.py    # ✅ Clean model interface (REWRITTEN)
│   ├── earthrover_interface.py  # ✅ SDK HTTP client (existing)
│   ├── dbr_module.py            # For DBR integration
│   ├── depth_estimator.py       # For DBR integration
│   └── depth_safety.py          # For DBR integration
├── scripts/
│   ├── navigate_citywalker.py   # ✅ Main navigation script (REWRITTEN)
│   ├── autonomous_exploration.py # Working exploration script
│   ├── archive/                 # Old scripts kept as reference
│   └── debug/                   # Debug/test utilities
├── samples/
│   ├── sample1.mp4
│   └── sample2.mp4
├── configs/
├── docs/
├── README.md
├── WHAT_WENT_WRONG.md       # Documents all mistakes made
├── CONVERSATION_LOG.md      # This file
└── environment.yml
```

---

## Key Technical Details

### CityWalker Model
- **Architecture:** 214M params, DINOv2 ViT-B/14 backbone + transformer decoder
- **Input:** 5 past frames (uint8 RGB) + 6 coordinates (5 past + 1 target)
- **Output:** 5 future waypoints (cumulative deltas) + arrival probability
- **Image preprocessing:** Model handles internally — ImageNet norm, center crop [350,630], resize [350,630]
- **Coordinate normalization:** Divide by step_scale on input, multiply on output

### Coordinate Transform (from teleop_dataset.py)
```python
# 1. GPS → local ENU meters
def latlon_to_local(lat, lon, ref_lat, ref_lon):
    R_earth = 6378137
    dlat = np.radians(lat - ref_lat)
    dlon = np.radians(lon - ref_lon)
    x = dlon * np.cos(np.radians((lat + ref_lat) / 2)) * R_earth
    y = dlat * R_earth
    return x, y

# 2. Transform to robot frame
def transform_input(positions):
    # Translate so current = origin
    current = positions[-1]
    translated = positions - current
    # Rotate so movement direction = +Y
    second_last = translated[-2]
    angle = -np.pi/2 - np.arctan2(second_last[1], second_last[0])
    rotation_matrix = [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
    rotated = translated @ rotation_matrix.T
    return rotated, angle

# 3. step_scale from trajectory
step_scale = np.linalg.norm(np.diff(positions, axis=0), axis=1).mean()
```

### EarthRover Interface
- SDK: FastAPI server on localhost:8000
- Client: `src/earthrover_interface.py` → `EarthRoverInterface`
- `rover.get_camera_frame()` → numpy (H,W,3) RGB uint8
- `rover.get_pose()` → (lat, lon, heading_radians)
- `rover.send_control(linear, angular)` → values in [-1, 1]

---

## Next Steps
1. **Test on actual robot** — All code is ready, need real GPS data
2. **DBR integration** — Depth Barrier Regularization for obstacle avoidance using `DBR/` directory
3. **Fine-tuning** — Use `finetune.yaml` config with teleop data collected from the robot
