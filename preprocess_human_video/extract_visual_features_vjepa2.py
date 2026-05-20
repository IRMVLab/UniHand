"""
Repository: https://github.com/IRMVLab/UniHand
Paper: Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
Authors: Ma et.al.

This file provides utilities for extracting V-JEPA2 visual features for RGB frame datasets.
"""

"""
Repository: https://github.com/IRMVLab/UniHand
Paper: Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
Authors: Ma et.al.

This file provides utilities for extracting V-JEPA2 visual features for RGB frame datasets.
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm


VJEPA2_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "vjepa2")
sys.path.insert(0, VJEPA2_DIR)

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_giant_xformers_rope


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    """Load local V-JEPA2 PyTorch checkpoint weights into the encoder."""
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["encoder"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print(f"Loaded pretrained weights from {pretrained_weights}. Load message: {msg}")


def build_pt_video_transform(img_size):
    """Build the same preprocessing pipeline used by the local PyTorch V-JEPA2 demo."""
    short_side_size = int(256.0 / 224 * img_size)
    return video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def tensor_to_feature_vector(features):
    """Mean-pool V-JEPA2 patch and time tokens into one vector per clip."""
    if not torch.is_tensor(features):
        if hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state
        elif isinstance(features, (tuple, list)):
            features = features[0]
        else:
            raise TypeError(f"Unsupported V-JEPA2 output type: {type(features)}")

    while features.ndim > 2:
        features = features.mean(dim=1)
    return features


def load_rgb_npy_image(image_path):
    """Load an RGB frame stored as a NumPy array."""
    image = np.load(image_path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 image, got shape {image.shape} from {image_path}")
    return image.astype(np.uint8, copy=False)


def extract_frame_feature(image_rgb, video_transform, model, device, clip_frames):
    """Extract one V-JEPA2 feature vector from a single RGB frame.

    V-JEPA2 is a video encoder, so a single image is repeated into a static
    clip before inference.
    """
    video = np.repeat(image_rgb[None, ...], clip_frames, axis=0)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)
    pixel_values = video_transform(video).unsqueeze(0).to(device)

    with torch.inference_mode():
        features = model(pixel_values)
        visual_feature = tensor_to_feature_vector(features)

    return visual_feature.cpu().numpy().astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract V-JEPA2 visual features from RGB .npy frame sequences.")
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Root directory containing multiple date folders, each with an 'rgb' folder.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Folder where extracted feature .npy files will be saved.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Sampling interval for frames (default: 1).",
    )
    parser.add_argument(
        "--pt_model_path",
        type=str,
        default="./weights/vitg-384.pt",
        help="Path to the local V-JEPA2 PyTorch checkpoint.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="Input image size used by the V-JEPA2 encoder.",
    )
    parser.add_argument(
        "--clip_frames",
        type=int,
        default=2,
        help="Number of repeated frames per static clip.",
    )
    return parser.parse_args()


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_rgb_frame(path):
    """Load an RGB frame stored as a NumPy array."""
    return load_rgb_npy_image(path)


def extract_features_from_image(model, video_transform, np_img, device, clip_frames):
    """Convert a NumPy image to a static clip, run V-JEPA2, and return a pooled feature."""
    return extract_frame_feature(
        image_rgb=np_img,
        video_transform=video_transform,
        model=model,
        device=device,
        clip_frames=clip_frames,
    )


def process_date_folder(date_path, model, video_transform, output_path, interval, device, clip_frames):
    """Process one date folder and extract V-JEPA2 features from all RGB frames."""
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
        if not os.path.isfile(rgb_path) or not rgb_file.lower().endswith(".npy"):
            continue

        try:
            frame = load_rgb_frame(rgb_path)
        except Exception as exc:
            print(f"Failed to load frame, skipping: {rgb_path}. Error: {exc}")
            continue

        feat = extract_features_from_image(model, video_transform, frame, device, clip_frames)
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

    print(f"Loading V-JEPA2 PyTorch model from: {args.pt_model_path}")
    print(f"V-JEPA2 config: image_size={args.img_size}, clip_frames={args.clip_frames}")
    model = vit_giant_xformers_rope(img_size=(args.img_size, args.img_size), num_frames=args.clip_frames)
    load_pretrained_vjepa_pt_weights(model, args.pt_model_path)
    model = model.to(device).eval()
    video_transform = build_pt_video_transform(img_size=args.img_size)

    date_folders = sorted(os.listdir(args.input_root))

    for date_name in date_folders:
        date_path = os.path.join(args.input_root, date_name)

        if not os.path.isdir(date_path):
            continue

        output_path = os.path.join(args.output_root, f"{date_name}.npy")

        if os.path.exists(output_path):
            print(f"Features already exist: {output_path}. Skipping.")
            continue

        print(f"Extracting features for: {date_name}")
        process_date_folder(
            date_path,
            model,
            video_transform,
            output_path,
            args.interval,
            device,
            args.clip_frames,
        )


if __name__ == "__main__":
    main()
