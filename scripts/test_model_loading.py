"""
Test script to verify CityWalker model loads and runs correctly.
Run this from the rover/ directory:
    conda activate rover
    python scripts/test_model_loading.py
"""

import sys
import os

# Add CityWalker to path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CityWalker'))

import torch
from omegaconf import OmegaConf


# This class is needed because the checkpoint was saved with it
class DictNamespace:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(**value))
            else:
                setattr(self, key, value)


def main():
    print("=" * 60)
    print("CityWalker Model Loading Test")
    print("=" * 60)

    # Step 1: Check CUDA availability
    print("\n[Step 1] Checking CUDA...")
    if torch.cuda.is_available():
        print(f"  CUDA is available: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("  CUDA not available, using CPU (will be slow)")
        device = torch.device("cpu")

    # Step 2: Load config
    print("\n[Step 2] Loading config...")
    config_path = os.path.join(os.path.dirname(__file__), '..', 'CityWalker', 'config', 'citywalk_2000hr.yaml')
    cfg = OmegaConf.load(config_path)
    print(f"  Config loaded from: {config_path}")
    print(f"  Model type: {cfg.model.type}")
    print(f"  Context size: {cfg.model.obs_encoder.context_size}")

    # Step 3: Load model
    print("\n[Step 3] Loading model from checkpoint...")
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'CityWalker_2000hr.ckpt')

    if not os.path.exists(checkpoint_path):
        print(f"  ERROR: Checkpoint not found at {checkpoint_path}")
        print("  Please download it first (see README.md)")
        return

    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Loading... (this may take a minute)")

    from pl_modules.citywalker_feat_module import CityWalkerFeatModule
    model = CityWalkerFeatModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    model.eval()
    model.to(device)
    print("  Model loaded successfully!")

    # Step 4: Create dummy input
    print("\n[Step 4] Creating dummy input...")
    batch_size = 1
    context_size = cfg.model.obs_encoder.context_size  # 5 past frames
    height, width = 480, 640  # Typical camera resolution

    # Dummy images: (batch, num_frames, channels, height, width)
    dummy_images = torch.randn(batch_size, context_size, 3, height, width).to(device)
    print(f"  Images shape: {dummy_images.shape}")
    print(f"    - {batch_size} batch")
    print(f"    - {context_size} frames")
    print(f"    - 3 RGB channels")
    print(f"    - {height}x{width} resolution")

    # Dummy coordinates: (batch, context_size + 1, 2)
    # 5 past positions + 1 target position
    num_coords = context_size + 1
    dummy_coords = torch.randn(batch_size, num_coords, 2).to(device)
    print(f"  Coordinates shape: {dummy_coords.shape}")
    print(f"    - {context_size} past positions + 1 target = {num_coords} total")
    print(f"    - 2 values each (x, y)")

    # Step 5: Run inference
    print("\n[Step 5] Running inference...")
    with torch.no_grad():
        waypoints, arrival, feature_pred, future_obs_enc = model(
            dummy_images,
            dummy_coords,
            future_obs=None  # Not needed for inference
        )
    print("  Inference completed!")

    # Step 6: Check outputs
    print("\n[Step 6] Output shapes:")
    print(f"  Waypoints: {waypoints.shape}")
    print(f"    - {waypoints.shape[1]} future positions predicted")
    print(f"    - Each is (x, y) coordinate")

    print(f"  Arrival logits: {arrival.shape}")
    arrival_prob = torch.sigmoid(arrival)
    print(f"    - Probability (after sigmoid): {arrival_prob.item():.4f}")

    # Step 7: Show sample output values
    print("\n[Step 7] Sample output values:")
    print(f"  Predicted waypoints (normalized):")
    for i, wp in enumerate(waypoints[0]):
        print(f"    Step {i+1}: x={wp[0].item():.4f}, y={wp[1].item():.4f}")

    print("\n" + "=" * 60)
    print("SUCCESS! Model loads and runs correctly.")
    print("=" * 60)
    print("\nNext step: Connect real camera data from EarthRover")


if __name__ == "__main__":
    main()
