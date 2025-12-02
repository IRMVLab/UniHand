"""
Repository: https://github.com/IRMVLab/UniHand
Paper: Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
Authors: Ma et.al.

This file contains the main pipeline for generating 3D hand trajectories.
"""



import os
import argparse
import numpy as np
import cv2
import pickle
import open3d as o3d



def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 3D hand keypoints from (2D+depth) and generate point clouds."
    )
    parser.add_argument("--input_root", type=str, required=True,
                        help="Root directory containing dated folders with depth/rgb images.")
    parser.add_argument("--keypoint_root", type=str, required=True,
                        help="Root directory containing 2D keypoints npy per date folder.")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output directory to save 3D trajectories.")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_keypoints(keypoint_path):
    keypoints = np.load(keypoint_path)[:, :2]
    if keypoints.shape[0] != 21:
        raise ValueError(f"Expected 21 keypoints, got {keypoints.shape[0]}")
    return keypoints


def load_depth(depth_path):
    return np.load(depth_path)


def compute_3d_keypoints(keypoints, depth_map):
    depth_t = depth_map.T
    max_x, max_y = depth_t.shape[0] - 1, depth_t.shape[1] - 1

    depths = np.array([
        depth_t[
            np.clip(int(x), 0, max_x),
            np.clip(int(y), 0, max_y)
        ]
        for (x, y) in keypoints
    ])
    return np.hstack((keypoints, depths.reshape(-1, 1))) 


def visualize_keypoints(keypoints, depths, depth_map, save_path):
    depth_img = cv2.normalize(depth_map, None, 0, 255,
                              cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)

    for (x, y), d in zip(keypoints[[0, 4, 8]], depths[[0, 4, 8]]):
        cv2.circle(depth_img, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(depth_img, f"{d:.2f}", (int(x) + 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(save_path, depth_img)



def create_point_cloud(rgb_image, depth_image, intrinsics):
    depth_image = depth_image / 1000.0
    depth_image[depth_image > 1.0] = 0

    rgb_image = rgb_image[..., ::-1]

    rgb_o3d = o3d.geometry.Image(rgb_image.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=1.0,
        convert_rgb_to_intensity=False,
    )

    intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(
        intrinsics['w'], intrinsics['h'],
        intrinsics['fx'], intrinsics['fy'],
        intrinsics['ox'], intrinsics['oy']
    )

    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_matrix)



def main():
    args = parse_args()

    intrinsics = {
        'fx': 745.77929688,
        'fy': 745.3269043,
        'ox': 628.28448486,
        'oy': 364.4375,
        'w': 1280,
        'h': 720
    }

    for date_name in os.listdir(args.input_root):
        date_input_path = os.path.join(args.input_root, date_name)
        if not os.path.isdir(date_input_path):
            continue

        print(f"\nProcessing date folder: {date_name}")

        depth_dir = os.path.join(date_input_path, "depth")
        rgb_dir = os.path.join(date_input_path, "rgb")
        keypoint_dir = os.path.join(args.keypoint_root, date_name)

        output_traj_dir = os.path.join(args.output_root, date_name, "traj3d_in_cam_keypoints")
        output_pc_dir = os.path.join(args.input_root, date_name)
        ensure_dir(output_traj_dir)
        ensure_dir(output_pc_dir)

        # --------------------------------------------
        # Step 1: Generate 3D Keypoint Trajectories
        # --------------------------------------------
        key_files = sorted([f for f in os.listdir(keypoint_dir)
                            if f.endswith("right_keypoints.npy")])

        traj_3d_dict = {}

        for kf in key_files:
            key_path = os.path.join(keypoint_dir, kf)
            depth_path = os.path.join(depth_dir, kf.replace("_right_keypoints.npy", ".npy"))

            keypoints_2d = load_keypoints(key_path)
            depth_map = load_depth(depth_path)
            keypoints_3d = compute_3d_keypoints(keypoints_2d, depth_map)

            frame_idx = int(kf.split("_")[0])
            traj_3d_dict[frame_idx] = keypoints_3d

            vis_path = os.path.join(output_traj_dir, kf.replace(".npy", "_vis.jpg"))
            visualize_keypoints(keypoints_2d, keypoints_3d[:, 2], depth_map, vis_path)

        traj_pkl = os.path.join(output_traj_dir, "traj3d_in_cam_dict.pkl")
        with open(traj_pkl, "wb") as f:
            pickle.dump(traj_3d_dict, f)

        print(f"Saved 3D keypoints: {traj_pkl}")

        # --------------------------------------------
        # Step 2: Create and Save Point Cloud (single frame or multi-frame)
        # --------------------------------------------
        for frame_file in os.listdir(depth_dir):
            if not frame_file.endswith(".npy"):
                continue

            frame_idx = frame_file.replace(".npy", "")
            depth_img = np.load(os.path.join(depth_dir, frame_file))
            rgb_path = os.path.join(rgb_dir, f"{frame_idx}.npy")

            if not os.path.exists(rgb_path):
                print(f"Missing RGB for frame {frame_idx}, skipping point cloud.")
                continue

            rgb_img = np.load(rgb_path)
            pc = create_point_cloud(rgb_img, depth_img, intrinsics)

            output_ply = os.path.join(output_pc_dir, f"{date_name}_point_cloud.ply")
            if frame_idx == "000000":
                # print(output_ply)
                o3d.io.write_point_cloud(output_ply, pc)

        print(f"Finished point cloud generation for {date_name}")


if __name__ == "__main__":
    main()
