"""
UniHand hand keypoint extraction (SAM 3D Body Hand branch only, output in crop coordinates)

Output files (per frame, per person, right hand):
- {frame}_p{id}_right_hand_keypoints_crop.npy: shape (21, 2), raw model keypoints (crop or full-image coords depending on model).
- {frame}_p{id}_right_hand_keypoints_full.npy: shape (21, 2), unified to full-image pixel coords (x, y) for downstream use.
- {frame}_p{id}_right_hand_crop.jpg: hand crop image 256x256, by default with 21 keypoints and indices drawn (use --no_draw_on_crop to disable).
- {frame}_p{id}_right_hand_on_full.jpg: keypoints drawn on full image (requires --draw_on_full).
- {frame}_p{id}_right_hand_bbox.npy: rhand_bbox [x1,y1,x2,y2] in full-image coords (requires --save_bbox).

Notes:
- Hand branch output is in MHR order (thumb tip -> wrist etc.); script reorders to COCO-wholebody (0=wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky), aligned with hand_keypoints.
- SAM 3D Body hand crop size is from config MODEL.IMAGE_SIZE (e.g. 256x256 or 256x192 WxH), not fixed 256x256.
- Script maps keypoints to 256x256 crop for drawing; npy coords match model internal crop resolution.

python extract_hand_keypoints_sam3d_hand_branch.py \
  --img_folder /path/to/images \
  --out_folder /path/to/hand_keypoints_sam3d_crop \
  --sam3d_checkpoint /path/to/sam_3d_body.pt \
  --sam3d_mhr_path /path/to/mhr_model.pt \
  --draw_on_crop \
  --save_bbox
"""

from pathlib import Path
import sys
import argparse
import os

_work_dir = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(_work_dir / ".cache" / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(_work_dir / ".cache"))
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
SAM3D_ROOT = REPO_ROOT / "sam-3d-body"
if str(SAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3D_ROOT))

import numpy as np
import cv2

HAND_CROP_W = 256
HAND_CROP_H = 256

# -----------------------------------------------------------------------------
# MHR70 (SAM 3D Body Hand branch) -> COCO-wholebody hand 21 keypoint index mapping
# Hand branch outputs rhand 21 points in MHR order: local 0=MHR21(thumb tip), 1=MHR22, ..., 20=MHR41(wrist).
# COCO order: 0=wrist, 1-4=thumb root->tip, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky.
# COCO index i corresponds to MHR local index MHR_RIGHT_TO_COCO[i]; after reorder, matches hand_keypoints.
# -----------------------------------------------------------------------------
MHR_RIGHT_TO_COCO = np.array(
    [
        20,  # COCO 0 (wrist)              <- MHR local 20 (MHR41)
        3, 2, 1, 0,   # COCO 1-4 (thumb 1..4) <- MHR 24,23,22,21 (thumb root->tip)
        7, 6, 5, 4,   # COCO 5-8 (index 1..4) <- MHR 28..25
        11, 10, 9, 8, # COCO 9-12 (middle 1..4)
        15, 14, 13, 12, # COCO 13-16 (ring 1..4)
        19, 18, 17, 16, # COCO 17-20 (pinky 1..4)
    ],
    dtype=np.int32,
)


def mhr_hand_to_coco_order(kp_mhr_21: np.ndarray) -> np.ndarray:
    """Reorder 21 points from Hand branch MHR order to COCO-wholebody order, aligned with hand_keypoints."""
    kp = np.asarray(kp_mhr_21, dtype=np.float64)
    if kp.ndim == 1:
        kp = kp.reshape(-1, 2)
    kp = kp[:, :2] if kp.shape[1] >= 2 else kp
    return kp[MHR_RIGHT_TO_COCO].copy()


def keypoints_to_draw_space(kp_crop, out_w=HAND_CROP_W, out_h=HAND_CROP_H, model_crop_w=None, model_crop_h=None):
    """
    Map model output hand keypoints to the draw target crop image coords (out_w x out_h, default 256x256).
    model_crop_w, model_crop_h: if model hand crop size is known (e.g. config MODEL.IMAGE_SIZE 512x512),
    pass them for scaling; otherwise infer from keypoint range (256x256 / 256x192 / 192x256 etc.).
    """
    kp = np.asarray(kp_crop, dtype=np.float64)
    if kp.ndim == 1:
        kp = kp.reshape(-1, 2)
    kp = kp[:, :2].copy()
    x_min, x_max = kp[:, 0].min(), kp[:, 0].max()
    y_min, y_max = kp[:, 1].min(), kp[:, 1].max()
    # Normalized coords: roughly in [0,1] or [-0.5, 0.5]
    if x_max <= 1.5 and x_min >= -0.5 and y_max <= 1.5 and y_min >= -0.5:
        if x_min >= 0 and y_min >= 0:
            kp[:, 0] = kp[:, 0] * out_w
            kp[:, 1] = kp[:, 1] * out_h
        else:
            kp[:, 0] = (kp[:, 0] + 0.5) * out_w
            kp[:, 1] = (kp[:, 1] + 0.5) * out_h
        return kp
    # Pixel coords: prefer configured model crop size (e.g. 512x512), else infer from range
    if model_crop_w is not None and model_crop_h is not None and model_crop_w > 0 and model_crop_h > 0:
        kp[:, 0] = kp[:, 0] * (out_w / float(model_crop_w))
        kp[:, 1] = kp[:, 1] * (out_h / float(model_crop_h))
        return kp
    eps = 1.0
    if y_max <= 192 + eps and y_max > 1 and x_max > 192:
        kp[:, 0] = kp[:, 0] * (out_w / 256.0)
        kp[:, 1] = kp[:, 1] * (out_h / 192.0)
    elif x_max <= 192 + eps and x_max > 1 and y_max > 192:
        kp[:, 0] = kp[:, 0] * (out_w / 192.0)
        kp[:, 1] = kp[:, 1] * (out_h / 256.0)
    else:
        if x_max > out_w or y_max > out_h:
            kp[:, 0] = kp[:, 0] * (out_w / max(x_max, 1))
            kp[:, 1] = kp[:, 1] * (out_h / max(y_max, 1))
    return kp


def draw_keypoints_with_indices(image, keypoints, color=(0, 255, 0), radius=6, font_scale=0.55):
    """Draw keypoints on image with indices 0–20 (easier to see on 256x256 hand crop)."""
    img = image.copy()
    kp = np.asarray(keypoints)
    if kp.ndim == 1:
        kp = kp.reshape(-1, 2)
    if kp.shape[1] >= 3:
        kp = kp[:, :2]
    h, w = img.shape[:2]
    for i in range(min(21, len(kp))):
        x, y = int(round(kp[i, 0])), int(round(kp[i, 1]))
        # Clamp to image bounds so points are visible when coords are in wrong space
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        cv2.circle(img, (x, y), radius, color, 2)
        cv2.putText(img, str(i), (x - 6, y - radius - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
    return img


def full_image_to_crop_keypoints(kp_full, bbox, out_w=HAND_CROP_W, out_h=HAND_CROP_H):
    """
    Convert keypoints from full-image coords to crop image coords (bbox cropped and resized to out_w x out_h).
    bbox: [x1, y1, x2, y2] in full-image pixels.
    """
    kp = np.asarray(kp_full, dtype=np.float64)[:, :2].copy()
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    bw = max(x2 - x1, 1e-6)
    bh = max(y2 - y1, 1e-6)
    # Relative position in bbox [0,1], then scale to out_w x out_h
    kp[:, 0] = (kp[:, 0] - x1) / bw * out_w
    kp[:, 1] = (kp[:, 1] - y1) / bh * out_h
    return kp


def crop_keypoints_to_full_image(kp_crop, bbox, model_crop_w, model_crop_h):
    """
    Convert keypoints from hand crop pixel coords to full-image coords.
    bbox: [x1, y1, x2, y2] full-image pixels; model_crop_w/h are model crop size (e.g. 512).
    """
    kp = np.asarray(kp_crop, dtype=np.float64)[:, :2].copy()
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    bw = max(x2 - x1, 1e-6)
    bh = max(y2 - y1, 1e-6)
    w = max(model_crop_w, 1)
    h = max(model_crop_h, 1)
    kp[:, 0] = x1 + (kp[:, 0] / w) * bw
    kp[:, 1] = y1 + (kp[:, 1] / h) * bh
    return kp


def _debug_keypoints(kp_crop, model_crop_w, model_crop_h, out_w, out_h, img_fn, person_id):
    """Print first-frame raw keypoint coords, scaled coords and scale factors for debugging x/y swap or scale errors."""
    kp = np.asarray(kp_crop, dtype=np.float64)[:, :2]
    c0_min, c0_max = kp[:, 0].min(), kp[:, 0].max()
    c1_min, c1_max = kp[:, 1].min(), kp[:, 1].max()
    print("  [DEBUG] frame=%s p%d raw keypoints (21,2):" % (img_fn, person_id))
    print("    col0 (treat as x): min=%.3f max=%.3f  col1 (treat as y): min=%.3f max=%.3f" % (c0_min, c0_max, c1_min, c1_max))
    print("    first 5 points (col0, col1): %s" % [tuple(kp[i].round(2)) for i in range(min(5, len(kp)))])
    if model_crop_w and model_crop_h:
        scale_x = out_w / float(model_crop_w)
        scale_y = out_h / float(model_crop_h)
        print("    scale: model_crop %dx%d -> out %dx%d  (scale_x=%.4f, scale_y=%.4f)" % (
            model_crop_w, model_crop_h, out_w, out_h, scale_x, scale_y))
    kp_draw = keypoints_to_draw_space(kp_crop, out_w, out_h, model_crop_w=model_crop_w, model_crop_h=model_crop_h)
    d0_min, d0_max = kp_draw[:, 0].min(), kp_draw[:, 0].max()
    d1_min, d1_max = kp_draw[:, 1].min(), kp_draw[:, 1].max()
    print("    after scale -> draw space: col0 [%.2f, %.2f]  col1 [%.2f, %.2f]" % (d0_min, d0_max, d1_min, d1_max))
    print("    first 5 in draw space: %s" % [tuple(kp_draw[i].round(2)) for i in range(min(5, len(kp_draw)))])
    if (c0_max - c0_min) < 10 and (c1_max - c1_min) > 100:
        print("    [HINT] col0 varies little, col1 varies a lot -> model may output (y,x), try swapping columns")
    elif (c1_max - c1_min) < 10 and (c0_max - c0_min) > 100:
        print("    [HINT] col1 varies little, col0 varies a lot -> model may output (x,y) with one axis squeezed, or swap columns")


def crop_hand_from_image(img_bgr, hand_bbox, out_w=HAND_CROP_W, out_h=HAND_CROP_H):
    """Crop hand region from full image by hand_bbox and resize to (out_w, out_h). Returns BGR image."""
    x1, y1, x2, y2 = [int(round(x)) for x in hand_bbox]
    h_img, w_img = img_bgr.shape[:2]
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    crop = img_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser(description="SAM 3D Body Hand branch: output right-hand keypoints in crop image coords")
    parser.add_argument("--img_folder", type=str, default="images", help="Input root dir containing sequence/rgb/")
    parser.add_argument("--out_folder", type=str, default="hand_keypoints_sam3d_crop", help="Output root dir")
    parser.add_argument("--sam3d_checkpoint", type=str, default="", required=True, help="SAM 3D Body checkpoint path")
    parser.add_argument("--sam3d_detector_path", type=str, default=str(REPO_ROOT / "hamer"), help="ViTDet weight dir or .pkl path")
    parser.add_argument("--sam3d_mhr_path", type=str, default="", help="MoHR model path mhr_model.pt")
    parser.add_argument("--bbox_thresh", type=float, default=0.8, help="Human bbox confidence threshold")
    parser.add_argument("--file_type", nargs="+", default=["*.jpg", "*.png", "*.npy"], help="Input file types")
    parser.add_argument("--seq_order", type=str, default="name", choices=("raw", "name", "ctime", "mtime", "list"),
                        help="Sequence dir traversal order; default name = sort by dir name")
    parser.add_argument("--seq_list", type=str, default="")
    parser.add_argument("--draw_on_crop", action="store_true", help="(default on) Draw keypoints on crop and save _right_hand_crop.jpg")
    parser.add_argument("--no_draw_on_crop", action="store_true", help="Do not save crop image with keypoints, only npy")
    parser.add_argument("--save_bbox", action="store_true", help="Save rhand_bbox as _right_hand_bbox.npy")
    parser.add_argument("--debug", action="store_true", help="Print keypoint coords and scale debug info (first frame of run)")
    parser.add_argument("--swap_xy", action="store_true", help="Swap keypoint columns when drawing (try if model outputs y,x)")
    parser.add_argument("--draw_on_full", action="store_true", help="Also draw keypoints on full image (full-image coords), save _right_hand_on_full.jpg")
    args = parser.parse_args()

    if not args.sam3d_checkpoint:
        raise ValueError("Must specify --sam3d_checkpoint")

    import torch
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

    mhr_path = args.sam3d_mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.sam3d_detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    if detector_path:
        p = Path(detector_path)
        if p.suffix == ".pkl" or (p.is_file() if p.exists() else False):
            detector_path = str(p.parent)
    segmentor_path = os.environ.get("SAM3D_SEGMENTOR_PATH", "")

    sam3d_model, sam3d_cfg = load_sam_3d_body(args.sam3d_checkpoint, device=device, mhr_path=mhr_path)
    human_detector = None
    try:
        from tools.build_detector import HumanDetector
        human_detector = HumanDetector(name="vitdet", device=device, path=detector_path)
    except Exception as e:
        print("Detector load failed:", e)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=sam3d_model,
        model_cfg=sam3d_cfg,
        human_detector=human_detector,
        human_segmentor=None,
        fov_estimator=None,
    )

    hand_data_path = os.path.abspath(args.img_folder)
    out_data_path = os.path.abspath(args.out_folder)
    all_dirs = [d for d in os.listdir(hand_data_path) if os.path.isdir(os.path.join(hand_data_path, d))]
    print("Input root:", hand_data_path)
    print("Output root:", out_data_path, "(seq_order=%s)" % args.seq_order)
    # Parse model hand crop size for scaling keypoints to 256x256 for drawing
    model_crop_w, model_crop_h = None, None
    hand_crop_size = getattr(sam3d_cfg.MODEL, "IMAGE_SIZE", None)
    if hand_crop_size is not None:
        try:
            if hasattr(hand_crop_size, "__iter__") and not isinstance(hand_crop_size, str):
                model_crop_w, model_crop_h = int(hand_crop_size[0]), int(hand_crop_size[1])
            else:
                model_crop_w = model_crop_h = int(hand_crop_size)
            print("Hand crop size (MODEL.IMAGE_SIZE): %d x %d (W x H)" % (model_crop_w, model_crop_h))
        except Exception:
            print("Hand crop size (MODEL.IMAGE_SIZE):", hand_crop_size)
    else:
        print("Hand crop size (MODEL.IMAGE_SIZE): not set, using model default (e.g. 256x256 or 256x192)")
    print("Keypoint coords: in hand crop pixel space (not full-image coords).")
    print("Output crop image: 256x256 with 21 keypoints drawn; npy coords match model crop resolution.")

    if args.seq_order == "list" and args.seq_list:
        with open(args.seq_list, "r", encoding="utf-8") as f:
            order_names = [line.strip() for line in f if line.strip()]
        hand_date_list = [n for n in order_names if n in all_dirs and os.path.isdir(os.path.join(hand_data_path, n))]
    elif args.seq_order == "ctime":
        dir_paths = [(d, os.path.join(hand_data_path, d)) for d in all_dirs]
        hand_date_list = [d for d, _ in sorted(dir_paths, key=lambda x: os.path.getctime(x[1]))]
    elif args.seq_order == "mtime":
        dir_paths = [(d, os.path.join(hand_data_path, d)) for d in all_dirs]
        hand_date_list = [d for d, _ in sorted(dir_paths, key=lambda x: os.path.getmtime(x[1]))]
    elif args.seq_order == "name":
        hand_date_list = sorted(all_dirs)
    else:
        hand_date_list = all_dirs

    debug_printed_once = False
    full_image_coords_warned = False  # Warn once when full-image coords are first detected
    for hand_date in hand_date_list:
        out_folder = os.path.join(out_data_path, hand_date)
        os.makedirs(out_folder, exist_ok=True)
        img_folder = os.path.join(hand_data_path, hand_date, "rgb")
        if not os.path.isdir(img_folder):
            continue
        img_paths = sorted([p for ext in args.file_type for p in Path(img_folder).glob(ext)])
        n_saved_kp, n_saved_img = 0, 0
        print("Processing folder:", img_folder)

        for img_path in img_paths:
            suf = (img_path.suffix or "").lower()
            if suf == ".npy":
                img_cv2 = np.load(str(img_path))
            else:
                img_cv2 = cv2.imread(str(img_path))
            if img_cv2 is None or img_cv2.ndim != 3 or img_cv2.shape[2] != 3:
                continue
            img_rgb = img_cv2[:, :, ::-1].copy()

            outputs = estimator.process_one_image(
                img_rgb, bbox_thr=args.bbox_thresh, use_mask=False
            )
            if len(outputs) == 0:
                continue

            img_fn, _ = os.path.splitext(os.path.basename(img_path))
            person_id = 0
            for person_out in outputs:
                rhand_bbox = person_out.get("rhand_bbox")
                rhand_crop_kp = person_out.get("pred_keypoints_2d_rhand_crop")
                if rhand_bbox is None or rhand_crop_kp is None:
                    continue

                # Hand branch output is MHR order (local 0=thumb tip..20=wrist); reorder to COCO-wholebody (0=wrist, 1-4 thumb, 5-8 index, ...) to match hand_keypoints
                kp_raw = np.asarray(rhand_crop_kp, dtype=np.float32)[:, :2]  # (21, 2) MHR order
                kp_crop = mhr_hand_to_coco_order(kp_raw).astype(np.float32)   # (21, 2) COCO order
                np.save(
                    os.path.join(out_folder, f"{img_fn}_p{person_id}_right_hand_keypoints_crop.npy"),
                    kp_crop,
                )
                if n_saved_kp == 0:
                    print("  [First keypoint range] x: [%.2f, %.2f], y: [%.2f, %.2f]" % (
                        kp_crop[:, 0].min(), kp_crop[:, 0].max(), kp_crop[:, 1].min(), kp_crop[:, 1].max()))
                    if args.debug and not debug_printed_once:
                        _debug_keypoints(
                            kp_crop, model_crop_w, model_crop_h, HAND_CROP_W, HAND_CROP_H,
                            img_fn, person_id,
                        )
                        debug_printed_once = True
                n_saved_kp += 1
                if args.save_bbox:
                    np.save(
                        os.path.join(out_folder, f"{img_fn}_p{person_id}_right_hand_bbox.npy"),
                        np.asarray(rhand_bbox, dtype=np.float32),
                    )

                kp_for_draw = kp_crop[:, [1, 0]].copy() if args.swap_xy else kp_crop
                model_max = max(model_crop_w or 512, model_crop_h or 512)
                is_full_image_coords = (
                    kp_for_draw[:, 0].max() > model_max * 1.1 or kp_for_draw[:, 1].max() > model_max * 1.1
                )
                # 21 keypoints in full-image coords, save as npy for downstream use
                if is_full_image_coords:
                    kp_full = kp_for_draw.copy()
                else:
                    kp_full = crop_keypoints_to_full_image(
                        kp_for_draw, rhand_bbox, model_crop_w or 512, model_crop_h or 512,
                    )
                np.save(
                    os.path.join(out_folder, f"{img_fn}_p{person_id}_right_hand_keypoints_full.npy"),
                    np.asarray(kp_full, dtype=np.float32),
                )

                # By default draw keypoints on output crop and save (unless --no_draw_on_crop)
                if not args.no_draw_on_crop:
                    hand_crop_img = crop_hand_from_image(img_cv2, rhand_bbox, HAND_CROP_W, HAND_CROP_H)
                    if is_full_image_coords:
                        if not full_image_coords_warned:
                            print("  [Info] Keypoints appear to be full-image coords (max > %.0f), converting with rhand_bbox to crop for drawing." % (model_max * 1.1))
                            full_image_coords_warned = True
                        kp_draw = full_image_to_crop_keypoints(kp_for_draw, rhand_bbox, HAND_CROP_W, HAND_CROP_H)
                    else:
                        kp_draw = keypoints_to_draw_space(
                            kp_for_draw, HAND_CROP_W, HAND_CROP_H,
                            model_crop_w=model_crop_w, model_crop_h=model_crop_h,
                        )
                    vis = draw_keypoints_with_indices(hand_crop_img, kp_draw, color=(0, 255, 0))
                    cv2.imwrite(
                        os.path.join(out_folder, f"{img_fn}_p{person_id}_right_hand_crop.jpg"),
                        vis,
                    )
                    n_saved_img += 1

                # Draw keypoints on full image (full-image coords), save _right_hand_on_full.jpg
                if args.draw_on_full:
                    full_vis = draw_keypoints_with_indices(
                        img_cv2.copy(), kp_full, color=(0, 255, 0), radius=8, font_scale=0.6,
                    )
                    cv2.imwrite(
                        os.path.join(out_folder, f"{img_fn}_p{person_id}_right_hand_on_full.jpg"),
                        full_vis,
                    )
                person_id += 1
        if img_paths:
            print("  -> saved %d keypoint npy, %d crop jpg in %s" % (n_saved_kp, n_saved_img, out_folder))


if __name__ == "__main__":
    main()
