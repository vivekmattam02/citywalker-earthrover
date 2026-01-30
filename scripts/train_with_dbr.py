"""
Train CityWalker with DBR (Depth Barrier Regularization)

This script trains the CityWalker model with the DBR safety loss.
DBR penalizes waypoints predicted toward obstacles, making the model
learn safer navigation behaviors.

USAGE:
    # Train from scratch with DBR
    python scripts/train_with_dbr.py --data-dir /path/to/citywalk_data

    # Fine-tune existing model with DBR
    python scripts/train_with_dbr.py --checkpoint models/CityWalker_2000hr.ckpt

    # Quick test (1 epoch, small dataset)
    python scripts/train_with_dbr.py --quick-test

REQUIREMENTS:
    - CityWalk dataset with depth maps (precomputed or online)
    - GPU recommended (CPU works but slow)
    - Depth Anything V2 checkpoint (for online depth)

Author: Vivek Mattam
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CityWalker'))

from dbr_module import DBRModule
from citywalker_wrapper import CityWalkerWrapper


class NavigationDataset(Dataset):
    """
    Dataset for training CityWalker with DBR.

    Each sample contains:
        - 5 RGB frames (context)
        - 6 coordinate pairs (5 past + 1 target)
        - 5 ground truth future waypoints
        - 1 arrival label (0 or 1)
        - 1 depth map (for DBR loss)
    """

    def __init__(self, data_dir, depth_dir=None, split='train',
                 context_size=5, pred_steps=5):
        """
        Args:
            data_dir: Path to dataset directory
            depth_dir: Path to precomputed depth maps (optional)
            split: 'train' or 'val'
            context_size: Number of past frames (5)
            pred_steps: Number of future waypoints (5)
        """
        self.data_dir = data_dir
        self.depth_dir = depth_dir
        self.context_size = context_size
        self.pred_steps = pred_steps

        # Load dataset index
        self.samples = self._load_samples(split)
        print(f"Loaded {len(self.samples)} {split} samples")

    def _load_samples(self, split):
        """Load sample indices from dataset."""
        samples = []

        # Look for preprocessed data
        index_file = os.path.join(self.data_dir, f'{split}_index.npy')
        if os.path.exists(index_file):
            samples = np.load(index_file, allow_pickle=True).tolist()
            return samples

        # Look for video/pose directories
        video_dir = os.path.join(self.data_dir, 'videos')
        pose_dir = os.path.join(self.data_dir, 'poses')

        if os.path.exists(video_dir) and os.path.exists(pose_dir):
            # Build samples from video/pose pairs
            pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith('.txt')])
            for pf in pose_files:
                video_name = pf.replace('.txt', '')
                samples.append({
                    'video': os.path.join(video_dir, video_name + '.mp4'),
                    'pose': os.path.join(pose_dir, pf),
                    'name': video_name
                })

        if not samples:
            print(f"WARNING: No samples found in {self.data_dir}")
            print("Expected structure:")
            print("  data_dir/videos/*.mp4")
            print("  data_dir/poses/*.txt")
            print("  OR data_dir/train_index.npy")

        return samples

    def __len__(self):
        return max(len(self.samples), 1)  # At least 1 for testing

    def __getitem__(self, idx):
        """
        Returns a training sample.

        For now, returns dummy data if no real dataset is available.
        Replace with actual data loading for real training.
        """
        # TODO: Replace with actual data loading when dataset is available
        # This provides the correct shapes for testing the pipeline

        H, W = 480, 640

        # 5 RGB frames
        images = torch.randn(self.context_size, 3, H, W)

        # 6 coordinate pairs (5 past + 1 target)
        coordinates = torch.randn(self.context_size + 1, 2) * 0.5

        # 5 ground truth waypoints
        gt_waypoints = torch.randn(self.pred_steps, 2) * 0.3

        # Arrival label
        arrived = torch.tensor(0.0)

        # Depth map (H, W) in meters
        depth_map = torch.rand(H, W) * 10.0 + 0.5

        # Step scale (for converting normalized to metric)
        step_scale = torch.tensor(1.0)

        return {
            'images': images,
            'coordinates': coordinates,
            'gt_waypoints': gt_waypoints,
            'arrived': arrived,
            'depth_map': depth_map,
            'step_scale': step_scale
        }


class CityWalkerDBRTrainer:
    """
    Training loop for CityWalker + DBR.

    Combines the standard CityWalker losses with the DBR barrier loss.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"\n{'='*60}")
        print(f"CityWalker + DBR Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")

        # Load model
        print("\n[1] Loading CityWalker model...")
        self.model_wrapper = CityWalkerWrapper(device=str(self.device))
        self.model = self.model_wrapper.model

        # Unfreeze decoder for training
        for param in self.model.parameters():
            param.requires_grad = False
        # Only train the decoder
        if hasattr(self.model, 'model'):
            decoder = self.model.model
            if hasattr(decoder, 'decoder'):
                for param in decoder.decoder.parameters():
                    param.requires_grad = True
            if hasattr(decoder, 'waypoint_head'):
                for param in decoder.waypoint_head.parameters():
                    param.requires_grad = True
            if hasattr(decoder, 'arrival_head'):
                for param in decoder.arrival_head.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,}")

        # Initialize DBR module
        print("\n[2] Initializing DBR module...")
        self.dbr = DBRModule(
            num_bins=config['num_bins'],
            temperature=config['temperature'],
            crop_bottom_ratio=config['crop_bottom_ratio'],
            fov_horizontal=config['fov_horizontal'],
            margin=config['margin'],
            loss_weight=config['dbr_weight'],
            image_width=640,
            image_height=480
        ).to(self.device)
        print(f"  Bins: {config['num_bins']}, Margin: {config['margin']}m")
        print(f"  Temperature: {config['temperature']}, Weight: {config['dbr_weight']}")

        # Optimizer (only trainable params)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable_params, lr=config['lr'],
                                     weight_decay=config.get('weight_decay', 0.01))

        # Loss functions
        self.waypoint_loss_fn = nn.L1Loss()
        self.arrival_loss_fn = nn.BCEWithLogitsLoss()
        self.direction_loss_weight = config.get('direction_loss_weight', 5.0)

        # Scheduler
        self.scheduler = None  # Set after dataloader is created

        # Metrics
        self.train_losses = []
        self.val_losses = []

    def compute_direction_loss(self, pred_waypoints, gt_waypoints):
        """Compute direction loss (1 - cosine similarity of first waypoint)."""
        pred_dir = F.normalize(pred_waypoints[:, 0, :], dim=-1)
        gt_dir = F.normalize(gt_waypoints[:, 0, :], dim=-1)
        cos_sim = (pred_dir * gt_dir).sum(dim=-1)
        return (1 - cos_sim).mean()

    def train_step(self, batch):
        """Execute one training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Unpack batch
        images = batch['images'].to(self.device)
        coordinates = batch['coordinates'].to(self.device)
        gt_waypoints = batch['gt_waypoints'].to(self.device)
        arrived = batch['arrived'].to(self.device)
        depth_map = batch['depth_map'].to(self.device)
        step_scale = batch['step_scale'].to(self.device)

        # Forward pass
        waypoints_pred, arrival_logits, _, _ = self.model(images, coordinates, future_obs=None)

        # Standard CityWalker losses
        wp_loss = self.waypoint_loss_fn(waypoints_pred, gt_waypoints)
        dir_loss = self.compute_direction_loss(waypoints_pred, gt_waypoints)

        # Arrival loss
        if arrival_logits.dim() > 1:
            arrival_logits = arrival_logits.squeeze(-1)
        arrive_loss = self.arrival_loss_fn(arrival_logits, arrived)

        # DBR loss
        wp_metric = waypoints_pred * step_scale.unsqueeze(-1).unsqueeze(-1)
        dbr_loss, clearance = self.dbr(wp_metric, depth_map)

        # Total loss
        total_loss = (wp_loss +
                      arrive_loss +
                      self.direction_loss_weight * dir_loss +
                      dbr_loss)

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'total': total_loss.item(),
            'waypoint': wp_loss.item(),
            'direction': dir_loss.item(),
            'arrival': arrive_loss.item(),
            'dbr': dbr_loss.item(),
            'clearance': clearance.mean().item()
        }

    @torch.no_grad()
    def val_step(self, batch):
        """Execute one validation step."""
        self.model.eval()

        images = batch['images'].to(self.device)
        coordinates = batch['coordinates'].to(self.device)
        gt_waypoints = batch['gt_waypoints'].to(self.device)
        depth_map = batch['depth_map'].to(self.device)
        step_scale = batch['step_scale'].to(self.device)

        waypoints_pred, arrival_logits, _, _ = self.model(images, coordinates, future_obs=None)

        wp_loss = self.waypoint_loss_fn(waypoints_pred, gt_waypoints)

        wp_metric = waypoints_pred * step_scale.unsqueeze(-1).unsqueeze(-1)
        dbr_loss, clearance = self.dbr(wp_metric, depth_map)

        return {
            'waypoint': wp_loss.item(),
            'dbr': dbr_loss.item(),
            'clearance': clearance.mean().item()
        }

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        losses = {'total': [], 'waypoint': [], 'direction': [],
                  'arrival': [], 'dbr': [], 'clearance': []}

        for batch_idx, batch in enumerate(train_loader):
            step_losses = self.train_step(batch)

            for k, v in step_losses.items():
                losses[k].append(v)

            # Print progress
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"\r  Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {step_losses['total']:.4f} | "
                      f"WP: {step_losses['waypoint']:.4f} | "
                      f"DBR: {step_losses['dbr']:.4f} | "
                      f"Clear: {step_losses['clearance']:.2f}m",
                      end="", flush=True)

        print()
        return {k: np.mean(v) for k, v in losses.items()}

    def validate(self, val_loader):
        """Run validation."""
        losses = {'waypoint': [], 'dbr': [], 'clearance': []}

        for batch in val_loader:
            step_losses = self.val_step(batch)
            for k, v in step_losses.items():
                losses[k].append(v)

        return {k: np.mean(v) for k, v in losses.items()}

    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 40)

            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            print(f"  Train | Loss: {train_losses['total']:.4f} | "
                  f"WP: {train_losses['waypoint']:.4f} | "
                  f"DBR: {train_losses['dbr']:.4f}")

            # Validate
            val_losses = self.validate(val_loader)
            print(f"  Val   | WP: {val_losses['waypoint']:.4f} | "
                  f"DBR: {val_losses['dbr']:.4f} | "
                  f"Clear: {val_losses['clearance']:.2f}m")

            # Save best model
            val_total = val_losses['waypoint'] + val_losses['dbr']
            if val_total < best_val_loss:
                best_val_loss = val_total
                save_path = os.path.join(os.path.dirname(__file__), '..',
                                         'models', 'CityWalker_DBR_best.ckpt')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"  Saved best model (val_loss={val_total:.4f})")

            print()

        print(f"{'='*60}")
        print(f"Training complete! Best val loss: {best_val_loss:.4f}")
        print(f"Model saved to: models/CityWalker_DBR_best.ckpt")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Train CityWalker with DBR")

    # Data
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to training data directory')
    parser.add_argument('--depth-dir', type=str, default=None,
                        help='Path to precomputed depth maps')

    # Model
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to pretrained CityWalker checkpoint')

    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')

    # DBR
    parser.add_argument('--margin', type=float, default=0.5,
                        help='Safety margin τ in meters')
    parser.add_argument('--dbr-weight', type=float, default=1.0,
                        help='DBR loss weight λ_bar')
    parser.add_argument('--num-bins', type=int, default=32,
                        help='Number of angular bins')
    parser.add_argument('--temperature', type=float, default=20.0,
                        help='Soft-min temperature κ')

    # Debug
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test with dummy data (1 epoch)')

    args = parser.parse_args()

    # Config
    config = {
        'lr': args.lr,
        'num_bins': args.num_bins,
        'temperature': args.temperature,
        'crop_bottom_ratio': 0.6,
        'fov_horizontal': 90.0,
        'margin': args.margin,
        'dbr_weight': args.dbr_weight,
        'direction_loss_weight': 5.0,
    }

    # Create trainer
    trainer = CityWalkerDBRTrainer(config)

    # Create datasets
    if args.quick_test or args.data_dir is None:
        print("\n[3] Using dummy data (quick test mode)...")
        train_dataset = NavigationDataset("dummy", split='train')
        val_dataset = NavigationDataset("dummy", split='val')
    else:
        print(f"\n[3] Loading data from {args.data_dir}...")
        train_dataset = NavigationDataset(args.data_dir, args.depth_dir, split='train')
        val_dataset = NavigationDataset(args.data_dir, args.depth_dir, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Train
    epochs = 1 if args.quick_test else args.epochs
    trainer.train(train_loader, val_loader, num_epochs=epochs)


if __name__ == "__main__":
    main()
