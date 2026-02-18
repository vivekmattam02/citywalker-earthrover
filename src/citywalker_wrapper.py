"""
CityWalker Model Wrapper

Provides a clean interface to load and run the CityWalker navigation model.

Usage:
    from citywalker_wrapper import CityWalkerWrapper

    wrapper = CityWalkerWrapper()
    waypoints, arrived = wrapper.predict(images, coordinates, step_scale=1.0)
"""

import sys
import os

# Add CityWalker to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CityWalker'))

import torch
import numpy as np
from omegaconf import OmegaConf


# Required for loading the checkpoint (it was saved with this class)
# We need to register it in __main__ so pickle can find it
class DictNamespace:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(**value))
            else:
                setattr(self, key, value)


# Register DictNamespace in __main__ module so checkpoint can unpickle it
import __main__
__main__.DictNamespace = DictNamespace


class CityWalkerWrapper:
    """
    Wrapper for CityWalker navigation model.

    Handles model loading, preprocessing, and inference.

    From the CityWalker source code:
    - Model internally applies ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    - Model internally center-crops to [350, 630] and resizes to [350, 630] (pretrained config)
    - Input images just need to be 0-1 float tensors
    - Coordinates are (6, 2): 5 past positions + 1 target, divided by step_scale
    - Output waypoints are cumulative deltas in normalized space, multiply by step_scale for meters
    """

    def __init__(self, checkpoint_path=None, config_path=None, device=None):
        """
        Load the CityWalker model.

        Args:
            checkpoint_path: Path to model checkpoint. Defaults to models/CityWalker_2000hr.ckpt
            config_path: Path to config file. Defaults to CityWalker/config/citywalk_2000hr.yaml
            device: 'cuda' or 'cpu'. Defaults to cuda if available.
        """
        # Set default paths
        base_dir = os.path.join(os.path.dirname(__file__), '..')

        if checkpoint_path is None:
            checkpoint_path = os.path.join(base_dir, 'models', 'CityWalker_2000hr.ckpt')

        if config_path is None:
            config_path = os.path.join(base_dir, 'CityWalker', 'config', 'citywalk_2000hr.yaml')

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Loading CityWalker model on {self.device}...")

        # Load config
        self.cfg = OmegaConf.load(config_path)
        self.context_size = self.cfg.model.obs_encoder.context_size  # 5
        self.len_traj_pred = self.cfg.model.decoder.len_traj_pred    # 5

        # Load model
        from pl_modules.citywalker_feat_module import CityWalkerFeatModule
        self.model = CityWalkerFeatModule.load_from_checkpoint(checkpoint_path, cfg=self.cfg)
        self.model.eval()
        self.model.to(self.device)

        print("Model loaded successfully!")

    def predict(self, images, coordinates, step_scale):
        """
        Run inference on the model.

        Args:
            images: Camera frames as numpy array (5, H, W, 3), uint8 0-255.
                    Model handles ImageNet normalization and crop/resize internally.

            coordinates: Position coordinates as numpy array (6, 2).
                        5 past positions + 1 target, in LOCAL METERS.
                        These should already be transformed to robot frame
                        (translated so current=origin, rotated so forward=+Y).
                        NOT yet divided by step_scale (this function does that).

            step_scale: Average distance between consecutive positions (meters).
                       Computed from GPS trajectory: mean(||diff(positions)||).
                       Used to normalize inputs and denormalize outputs.
                       Typical values: 0.5-1.5m at 1Hz.

        Returns:
            waypoints: numpy array (5, 2) - 5 future (x, y) positions in METERS
            arrived: float 0-1 - probability that we've reached the goal
        """
        # Convert images to tensor
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
            images = images.permute(0, 3, 1, 2)  # (N, H, W, 3) -> (N, 3, H, W)
            images = images / 255.0  # To 0-1 (model applies ImageNet norm internally)

        # Convert coordinates to tensor
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.from_numpy(coordinates).float()

        # Normalize coordinates by step_scale (model trained with normalized coords)
        step_scale = max(step_scale, 0.01)  # Clamp to avoid division by zero
        coordinates = coordinates / step_scale

        # Add batch dimension
        images = images.unsqueeze(0)        # (1, N, 3, H, W)
        coordinates = coordinates.unsqueeze(0)  # (1, 6, 2)

        # Move to device
        images = images.to(self.device)
        coordinates = coordinates.to(self.device)

        # Run inference
        with torch.no_grad():
            waypoints, arrival_logits, _, _ = self.model(images, coordinates, future_obs=None)

        # Denormalize waypoints to get meters
        waypoints = waypoints[0].cpu().numpy()  # (5, 2)
        waypoints = waypoints * step_scale

        arrival_prob = torch.sigmoid(arrival_logits[0]).item()  # Scalar 0-1

        return waypoints, arrival_prob

    def get_context_size(self):
        """Returns how many past frames the model expects (5)."""
        return self.context_size

    def get_prediction_steps(self):
        """Returns how many future waypoints the model predicts (5)."""
        return self.len_traj_pred


# Quick test if run directly
if __name__ == "__main__":
    print("Testing CityWalkerWrapper...")

    wrapper = CityWalkerWrapper()

    # Create dummy data
    dummy_images = np.random.randint(0, 255, (5, 480, 640, 3), dtype=np.uint8)
    dummy_coords = np.random.randn(6, 2).astype(np.float32)

    # Run prediction with step_scale=1.0
    waypoints, arrived = wrapper.predict(dummy_images, dummy_coords, step_scale=1.0)

    print(f"\nResults:")
    print(f"  Waypoints shape: {waypoints.shape}")
    print(f"  Arrival probability: {arrived:.4f}")
    print(f"\n  Predicted waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"    Step {i+1}: x={wp[0]:.4f}, y={wp[1]:.4f}")

    print("\nWrapper test passed!")
