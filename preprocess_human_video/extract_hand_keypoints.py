"""
Repository: https://github.com/IRMVLab/UniHand
Paper: Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
Authors: Ma et.al.

This file performs per-frame hand mesh reconstruction using the HaMeR model,
runs body and hand keypoint detection, renders 3D MANO hand meshes back into
the original image frames, and saves both visualizations and mesh outputs.
"""


from pathlib import Path
import torch
import argparse
import os

# Use OSMesa for off-screen rendering
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional


def draw_keypoints_on_image(image, keypoints, color=(0, 255, 0), radius=5, thickness=-1):
    """
    Draw keypoints on an image.

    Args:
        image: Input image as (H, W, 3).
        keypoints: Array of keypoints with shape (N, 2) or (N, 3).
        color: Drawing color in BGR format.
        radius: Circle radius.
        thickness: Thickness (-1 = filled circle).

    Returns:
        Image with drawn keypoints.
    """
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(image, (x, y), radius, color, thickness)
    return image


def main():
    parser = argparse.ArgumentParser(description='HaMeR hand reconstruction demo')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder containing input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder for saving results')
    parser.add_argument('--side_view', action='store_true', default=False, help='Render additional side view')
    parser.add_argument('--full_frame', action='store_true', default=True, help='Render full-frame merged scene')
    parser.add_argument('--save_mesh', action='store_true', default=False, help='Save reconstructed meshes to disk')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='BBox padding/rescale factor')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Body detection model')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.npy'], help='Valid input file types')

    args = parser.parse_args()

    # Load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    hand_data_path = args.img_folder
    out_data_path = args.out_folder

    # Load human detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer

        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))

        detectron2_cfg.train.init_checkpoint = "./model_final_f05665.pkl"

        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25

        detector = DefaultPredictor_Lazy(detectron2_cfg)

    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg

        detectron2_cfg = model_zoo.get_config(
            'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4

        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Keypoint detector
    cpm = ViTPoseModel(device)

    # Renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    hand_date_list = os.listdir(hand_data_path)

    for hand_date in hand_date_list:

        out_folder = os.path.join(out_data_path, hand_date)
        if os.path.exists(out_folder):
            print("Skipping existing folder:", out_folder)
            continue

        os.makedirs(out_folder, exist_ok=True)

        img_folder = os.path.join(hand_data_path, hand_date, "rgb")
        img_paths = [img for end in args.file_type for img in Path(img_folder).glob(end)]
        img_paths = sorted(img_paths)

        print("Processing folder:", img_folder)

        for img_path in img_paths:

            print("Processing:", img_path)
            img_cv2 = np.load(str(img_path))

            # Human detection
            det_out = detector(img_cv2)
            img = img_cv2[:, :, ::-1]

            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores = det_instances.scores[valid_idx].cpu().numpy()

            # Human keypoint detection
            vitposes_out = cpm.predict_pose(
                img, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
            )

            bboxes, is_right, right_hand_keyp_all = [], [], []

            for vitposes in vitposes_out:
                left_hand_keyp = vitposes['keypoints'][-42:-21]
                right_hand_keyp = vitposes['keypoints'][-21:]

                right_hand_keyp_all.append(right_hand_keyp)

                # Left hand
                keyp = left_hand_keyp
                valid = keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [
                        keyp[valid, 0].min(),
                        keyp[valid, 1].min(),
                        keyp[valid, 0].max(),
                        keyp[valid, 1].max(),
                    ]
                    bboxes.append(bbox)
                    is_right.append(0)

                # Right hand
                keyp = right_hand_keyp
                valid = keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [
                        keyp[valid, 0].min(),
                        keyp[valid, 1].min(),
                        keyp[valid, 0].max(),
                        keyp[valid, 1].max(),
                    ]
                    bboxes.append(bbox)
                    is_right.append(1)

            if len(bboxes) == 0:
                print("No hands detected.")
                continue

            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            right_hand_keyp_all = np.stack(right_hand_keyp_all)

            # Reconstruction dataset
            dataset = ViTDetDataset(
                model_cfg, img_cv2, boxes, right, right_hand_keyp_all,
                rescale_factor=args.rescale_factor
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=8, shuffle=False, num_workers=0
            )

            all_verts, all_cam_t, all_right = [], [], []

            for batch in dataloader:
                batch = recursive_to(batch, device)

                with torch.no_grad():
                    out = model(batch)

                multiplier = (2 * batch['right'] - 1)
                pred_cam = out['pred_cam']
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]

                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()

                scaled_focal_length = (
                    model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE
                ) * img_size.max()

                pred_cam_t_full = cam_crop_to_full(
                    pred_cam, box_center, box_size, img_size, scaled_focal_length
                ).detach().cpu().numpy()

                batch_size = batch['img'].shape[0]

                for n in range(batch_size):
                    img_fn, _ = os.path.splitext(os.path.basename(img_path))
                    person_id = int(batch['personid'][n])

                    white_img = (
                        torch.ones_like(batch['img'][n]).cpu()
                        - DEFAULT_MEAN[:, None, None] / 255
                    ) / (DEFAULT_STD[:, None, None] / 255)

                    input_patch = (
                        batch['img'][n].cpu() * (DEFAULT_STD[:, None, None] / 255)
                        + (DEFAULT_MEAN[:, None, None] / 255)
                    )
                    input_patch = input_patch.permute(1, 2, 0).numpy()

                    regression_img = renderer(
                        out['pred_vertices'][n].detach().cpu().numpy(),
                        out['pred_cam_t'][n].detach().cpu().numpy(),
                        batch['img'][n],
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                    )

                    if args.side_view:
                        side_img = renderer(
                            out['pred_vertices'][n].detach().cpu().numpy(),
                            out['pred_cam_t'][n].detach().cpu().numpy(),
                            white_img,
                            mesh_base_color=LIGHT_BLUE,
                            scene_bg_color=(1, 1, 1),
                            side_view=True
                        )
                        final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                    else:
                        final_img = np.concatenate([input_patch, regression_img], axis=1)

                    cv2.imwrite(
                        os.path.join(out_folder, f'{img_fn}_{person_id}.png'),
                        255 * final_img[:, :, ::-1]
                    )

                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    is_r = batch['right'][n].cpu().numpy()
                    verts[:, 0] = (2 * is_r - 1) * verts[:, 0]

                    cam_t = pred_cam_t_full[n]

                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_r)

                    if args.save_mesh:
                        camera_translation = cam_t.copy()
                        tmesh = renderer.vertices_to_trimesh(
                            verts, camera_translation, LIGHT_BLUE, is_right=is_r
                        )
                        tmesh.export(os.path.join(out_folder, f'{img_fn}_{person_id}.obj'))

                right_keypoints_for_draw = batch['right_keypoints'][0]

            # Render merged scene
            if args.full_frame and len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )

                cam_view = renderer.render_rgba_multiple(
                    all_verts, cam_t=all_cam_t, render_res=img_size[n],
                    is_right=all_right, **misc_args
                )

                input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                input_img = np.concatenate(
                    [input_img, np.ones_like(input_img[:, :, :1])], axis=2
                )

                alpha = 0.3
                merged = (
                    input_img[:, :, :3] * (1 - alpha)
                    + cam_view[:, :, :3] * alpha
                )

                out_img = draw_keypoints_on_image(
                    merged, right_keypoints_for_draw, color=(0, 255, 0)
                )

                cv2.imwrite(os.path.join(out_folder, f'{img_fn}_all.jpg'),
                            255 * out_img[:, :, ::-1])

                cv2.imwrite(os.path.join(out_folder, f'{img_fn}_mask.jpg'),
                            255 * cam_view[:, :, :3])

                np.save(os.path.join(out_folder, f'{img_fn}_right_keypoints'),
                        right_keypoints_for_draw.cpu().numpy())


if __name__ == '__main__':
    main()
