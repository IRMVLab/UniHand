"""
Repository: https://github.com/IRMVLab/UniHand
Paper: Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
Authors: Ma et.al.

This file provides utilities for extracting DINOv2 visual features for RGB frame datasets.
"""


import os
import argparse
import numpy as np
import cv2
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract DINOv2 visual features for RGB frame datasets.")
    parser.add_argument("--input_root", type=str, required=True,
                        help="Root directory containing multiple date folders, each with an 'rgb' folder.")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Folder where extracted feature .npy files will be saved.")
    parser.add_argument("--interval", type=int, default=1,
                        help="Sampling interval for frames (default: 1).")
    return parser.parse_args()


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_rgb_frame(path):
    """Load an RGB frame stored as a numpy array."""
    return np.load(path)  # shape: (H, W, 3)


def extract_features_from_image(model, processor, np_img, device):
    """Convert numpy image to tensor, run DINOv2, return pooled feature."""
    pil_img = Image.fromarray(np_img)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs[0]                    # (B, tokens, C)
        pooled_feature = hidden_states.mean(dim=1)   # (B, C)

    return pooled_feature.cpu().numpy()  # return numpy vector


def process_date_folder(date_path, model, processor, output_path, interval, device):
    """Process one date folder and extract features from all RGB frames."""
    rgb_dir = os.path.join(date_path, "rgb")
    if not os.path.isdir(rgb_dir):
        print(f"Skipping {date_path}: missing 'rgb' folder.")
        return

    rgb_files = sorted(os.listdir(rgb_dir))
    features = []
    counter = 0

    for rgb_file in tqdm(rgb_files, desc=f"Processing {os.path.basename(date_path)}"):
        if counter % interval != 0:
            counter += 1
            continue
        counter += 1

        rgb_path = os.path.join(rgb_dir, rgb_file)
        frame = load_rgb_frame(rgb_path)
        feat = extract_features_from_image(model, processor, frame, device)
        features.append(feat)

    if len(features) == 0:
        print(f"No frames processed for {date_path}.")
        return

    features_array = np.concatenate(features, axis=0)
    ensure_dir(os.path.dirname(output_path))
    np.save(output_path, features_array)

    print(f"Saved feature array {features_array.shape} to {output_path}")


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load DINOv2 backbone
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

    # Get list of date folders
    date_folders = sorted(os.listdir(args.input_root))

    for date_name in date_folders:
        date_path = os.path.join(args.input_root, date_name)

        if not os.path.isdir(date_path):
            continue

        output_path = os.path.join(args.output_root, f"{date_name}.npy")

        # Skip if already processed
        if os.path.exists(output_path):
            print(f"Features already exist: {output_path}. Skipping.")
            continue

        print(f"Extracting features for: {date_name}")
        process_date_folder(date_path, model, processor, output_path, args.interval, device)


if __name__ == "__main__":
    main()
