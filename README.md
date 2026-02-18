# CityWalker-EarthRover Integration

Deploying [CityWalker](https://github.com/ai4ce/CityWalker) (CVPR 2025) navigation model on [FrodoBots EarthRover](https://github.com/frodobots-org/earth-rovers-sdk) for autonomous urban navigation.

## Overview

This project enables autonomous sidewalk navigation by integrating a vision-based navigation model with a mobile robot platform.

| Component | Description |
|-----------|-------------|
| CityWalker | Vision-based navigation model from AI4CE Lab, NYU |
| EarthRover Zero | Mobile robot with dual cameras, GPS, and IMU |

## Repository Structure

```
rover/
├── CityWalker/           # Navigation model repository
├── earth-rovers-sdk/     # Robot SDK
├── models/               # Pretrained weights (downloaded separately)
├── src/                  # Integration code
├── configs/              # Deployment configurations
├── PROGRESS.md           # Development tracker
└── environment.yml       # Conda environment
```

## Setup

### 1. Clone the repository
```bash
git clone <YOUR_REPO_URL>   # e.g. https://github.com/yourusername/rover.git
cd rover
```

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate rover
```

### 3. Download pretrained model (for CityWalker / outdoor nav)
```bash
mkdir -p models
wget -O models/CityWalker_2000hr.ckpt \
  "https://github.com/ai4ce/CityWalker/releases/download/v1.0/CityWalker_2000hr.ckpt"
```
*(Skip this if you only want to run Bug2 indoor exploration — that script doesn’t use CityWalker.)*

### 4. Configure the robot (SDK)
The SDK loads `.env` from **earth-rovers-sdk/** when you start the server. Create it there:

```bash
cp earth-rovers-sdk/.env.sample earth-rovers-sdk/.env
# Edit earth-rovers-sdk/.env and set:
# SDK_API_TOKEN=<your FrodoBots SDK token for this robot>
# BOT_SLUG=<your robot slug, e.g. jack-drax-hinge>
```

Minimum required in `earth-rovers-sdk/.env`:
```
SDK_API_TOKEN=your_token_here
BOT_SLUG=your_bot_slug_here
```

Get `SDK_API_TOKEN` and `BOT_SLUG` from the FrodoBots dashboard for your robot.

### 5. Verify installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Run a demo (after setup)

1. **Start the SDK server** (one terminal, leave it running):
   ```bash
   cd earth-rovers-sdk && hypercorn main:app --reload
   ```

2. **Open the browser**: go to **http://localhost:8000/sdk** in Chrome. Wait until you see live video and sensor data.

3. **Run a script** (another terminal, from project root):
   ```bash
   conda activate rover
   cd /path/to/rover

   # Indoor exploration (Bug2, no CityWalker, no GPS):
   python scripts/autonomous_exploration.py

   # Outdoor GPS + CityWalker:
   python scripts/outdoor_nav.py --target-lat <LAT> --target-lon <LON>

   # Indoor with keyboard + CityWalker:
   python scripts/indoor_nav.py
   ```

The browser tab must stay open and connected so the robot stream and commands work.

## Architecture

```
EarthRover          Integration Layer         CityWalker Model
    |                      |                        |
Camera frames  -->   Preprocessing    -->    DINOv2 Encoder
GPS/IMU data   -->   Coord Transform  -->    Transformer
Motor Control  <--   PD Controller    <--    Waypoint Prediction
```

## Requirements

- Python 3.11
- PyTorch 2.5+ with CUDA 12.1
- Conda

## Status

Work in progress. See [PROGRESS.md](PROGRESS.md) for details.

## References

- [CityWalker Paper](https://arxiv.org/abs/2411.17820)
- [CityWalker Repository](https://github.com/ai4ce/CityWalker)
- [EarthRover SDK](https://github.com/frodobots-org/earth-rovers-sdk)
