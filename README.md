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
git clone https://github.com/vivekmattam02/citywalker-earthrover.git
cd citywalker-earthrover
```

### 2. Download pretrained model
```bash
mkdir -p models
wget -O models/CityWalker_2000hr.ckpt \
  "https://github.com/ai4ce/CityWalker/releases/download/v1.0/CityWalker_2000hr.ckpt"
```

### 3. Create environment
```bash
conda env create -f environment.yml
conda activate rover
```

### 4. Verify installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

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
