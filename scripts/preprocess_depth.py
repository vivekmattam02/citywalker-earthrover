"""
Precompute Depth Maps for DBR Training

Processes video frames through Depth Anything V2 and saves depth maps
for efficient training with DBR.

USAGE:
    # Process a single video
    python scripts/preprocess_depth.py --input video.mp4 --output depths/

    # Process a directory of videos
    python scripts/preprocess_depth.py --input-dir data/videos/ --output-dir data/depths/

    # Process frames from robot camera (saved images)
    python scripts/preprocess_depth.py --input-dir data/frames/ --output-dir data/depths/

Author: Vivek Mattam
"""

import sys
import os
import argparse
import time
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def process_video(video_path, output_dir, estimator, fps_target=1, batch_size=4):
    """
    Process a video file and save depth maps.

    Args:
        video_path: Path to input video
        output_dir: Directory to save depth maps
        estimator: DepthEstimator instance
        fps_target: Target FPS for frame extraction
        batch_size: Batch size for depth inference
    """
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{video_name}_depth.npy")

    if os.path.exists(output_path):
        print(f"  Skipping {video_name} (already processed)")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps_target))

    print(f"  Processing {video_name}: {total_frames} frames at {video_fps:.1f} fps")
    print(f"  Extracting every {frame_interval} frames ({fps_target} fps target)")

    frames = []
    depth_maps = []
    frame_idx = 0
    extracted = 0

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            extracted += 1

            # Process in batches
            if len(frames) >= batch_size:
                batch_depths = []
                for f in frames:
                    depth = estimator.estimate(f)
                    batch_depths.append(depth)
                depth_maps.extend(batch_depths)
                frames = []

                # Progress
                elapsed = time.time() - start_time
                fps_proc = extracted / elapsed if elapsed > 0 else 0
                print(f"\r    Frame {frame_idx}/{total_frames} | "
                      f"Extracted: {extracted} | "
                      f"Speed: {fps_proc:.1f} frames/s", end="", flush=True)

        frame_idx += 1

    # Process remaining frames
    if frames:
        for f in frames:
            depth = estimator.estimate(f)
            depth_maps.append(depth)

    cap.release()

    if depth_maps:
        # Save as numpy array
        depth_array = np.stack(depth_maps, axis=0)  # (N, H, W)
        np.save(output_path, depth_array.astype(np.float16))  # float16 to save space

        elapsed = time.time() - start_time
        print(f"\n  Saved: {output_path}")
        print(f"  Shape: {depth_array.shape}, Size: {os.path.getsize(output_path)/1e6:.1f} MB")
        print(f"  Time: {elapsed:.1f}s")
    else:
        print(f"\n  WARNING: No frames extracted from {video_name}")


def process_image_directory(input_dir, output_dir, estimator):
    """
    Process a directory of images and save depth maps.

    Args:
        input_dir: Directory containing RGB images
        output_dir: Directory to save depth maps
        estimator: DepthEstimator instance
    """
    # Find all images
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if Path(f).suffix.lower() in extensions
    ])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(image_files)} images from {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, Path(img_file).stem + '_depth.npy')

        if os.path.exists(output_path):
            continue

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Estimate depth
        depth = estimator.estimate(img_rgb)

        # Save
        np.save(output_path, depth.astype(np.float16))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(image_files)} images")

    print(f"Done! Depth maps saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Precompute depth maps for DBR")

    parser.add_argument('--input', type=str, default=None,
                        help='Path to input video file')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Path to directory of videos or images')
    parser.add_argument('--output-dir', type=str, default='data/depths',
                        help='Output directory for depth maps')
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'base', 'large'],
                        help='Depth model size')
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Target FPS for video frame extraction')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for depth inference')

    args = parser.parse_args()

    if args.input is None and args.input_dir is None:
        parser.print_help()
        print("\n" + "=" * 60)
        print("USAGE EXAMPLES:")
        print("=" * 60)
        print("\n1. Process a single video:")
        print("   python scripts/preprocess_depth.py --input data/video.mp4")
        print("\n2. Process all videos in a directory:")
        print("   python scripts/preprocess_depth.py --input-dir data/videos/")
        print("\n3. Process saved frames:")
        print("   python scripts/preprocess_depth.py --input-dir data/frames/")
        return

    # Initialize depth estimator
    from depth_estimator import DepthEstimator
    estimator = DepthEstimator(model_size=args.model_size)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input:
        # Single video
        process_video(args.input, args.output_dir, estimator,
                      fps_target=args.fps, batch_size=args.batch_size)

    elif args.input_dir:
        # Check if it's videos or images
        files = os.listdir(args.input_dir)
        video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

        has_videos = any(Path(f).suffix.lower() in video_exts for f in files)
        has_images = any(Path(f).suffix.lower() in image_exts for f in files)

        if has_videos:
            # Process all videos
            video_files = sorted([
                f for f in files if Path(f).suffix.lower() in video_exts
            ])
            print(f"Found {len(video_files)} videos")
            for vf in video_files:
                video_path = os.path.join(args.input_dir, vf)
                process_video(video_path, args.output_dir, estimator,
                              fps_target=args.fps, batch_size=args.batch_size)
        elif has_images:
            # Process images
            process_image_directory(args.input_dir, args.output_dir, estimator)
        else:
            print(f"No videos or images found in {args.input_dir}")


if __name__ == "__main__":
    main()
