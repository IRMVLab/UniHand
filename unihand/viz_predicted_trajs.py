"""
Repository: https://github.com/IRMVLab/UniHand
Paper: Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
Authors: Ma et.al.

This file visualizes waypoints predicted by Uni-Hand, which an end-effector can follow.
"""

import os
import yaml
import numpy as np
import open3d as o3d


def load_config(yaml_path="./configs/traineval.yaml"):
    with open(yaml_path, "r") as f:
        cfg_traineval = yaml.safe_load(f)
        return cfg_traineval["viz"], cfg_traineval["data"]["data_root"], cfg_traineval["data"]["intrinsics"]


def pixel_to_cam(x_pixel, y_pixel, z_mm, intrinsics):
    z_m = z_mm / 1000.0
    x_cam = (x_pixel - intrinsics["ox"]) * z_m / intrinsics["fx"]
    y_cam = (y_pixel - intrinsics["oy"]) * z_m / intrinsics["fy"]
    return np.array([x_cam, y_cam, z_m])


def create_spheres(points, contact_idx, separate_idx, radius, highlight_radius):
    spheres = o3d.geometry.TriangleMesh()

    for i, point in enumerate(points):
        if i == contact_idx:
            color = [0, 1, 0]
            r = highlight_radius
        elif i == separate_idx:
            color = [1.0, 0.5, 0.7]
            r = highlight_radius
        elif i < contact_idx:
            color = [1, 0, 0]
            r = radius
        elif i < separate_idx:
            color = [0, 0, 1]
            r = radius
        else:
            color = [0.5, 0.5, 0.5]
            r = radius

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        sphere.paint_uniform_color(color)
        sphere.translate(point)
        spheres += sphere

    return spheres


def create_thick_lines(points, contact_idx, separate_idx, sphere_radius, density):
    meshes = o3d.geometry.TriangleMesh()

    for i in range(len(points) - 1):
        start, end = points[i], points[i + 1]
        direction = end - start
        length = np.linalg.norm(direction)
        if length == 0:
            continue

        if i < contact_idx:
            color = [1, 0, 0]
        elif i < separate_idx:
            color = [0, 0, 1]
        else:
            color = [0.5, 0.5, 0.5]

        for j in range(density):
            alpha = j / (density - 1)
            pos = start + alpha * direction

            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.paint_uniform_color(color)
            sphere.translate(pos)
            meshes += sphere

    return meshes


def main():
    cfg, data_root, intr = load_config()

    pred_base = cfg["pred_base"]
    contact_pred_path = cfg["contact_pred_path"]
    gt_base = cfg["gt_base"]
    contact_gt_path = cfg["contact_gt_path"]

    wrist_px = np.array(cfg["wrist_pixel"])
    grasp_px = np.array(cfg["grasp_pixel"])

    grasp_offset = (
        pixel_to_cam(*grasp_px, intr) - pixel_to_cam(*wrist_px, intr)
    )

    date_list = sorted(os.listdir(pred_base))

    for file_npy in date_list:
        date = file_npy[:-4]
        print(f"Processing {date}")

        rgb = np.load(os.path.join(data_root, date, "rgb/000000.npy"))
        depth = np.load(os.path.join(data_root, date, "depth/000000.npy"))
        wrist_pred = np.load(os.path.join(pred_base, file_npy))
        wrist_gt = np.load(os.path.join(gt_base, file_npy))

        contact_pred = np.load(os.path.join(contact_pred_path, file_npy))
        contact_idx = int(contact_pred[0])
        separate_idx = int(contact_pred[1])
        print(f"[Pred] contact={contact_idx}, separate={separate_idx}")

        contact_gt = np.load(os.path.join(contact_gt_path, file_npy))
        print(f"[GT]   contact={int(contact_gt[0])}, separate={int(contact_gt[1])}")

        grasp_pred = wrist_pred + grasp_offset

        intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(
            intr["w"], intr["h"], intr["fx"], intr["fy"], intr["ox"], intr["oy"]
        )

        rgb_o3d = o3d.geometry.Image(rgb[..., ::-1].astype(np.uint8))
        depth_m = (depth / 1000.0).astype(np.float32)
        depth_m[depth_m > 1.0] = 0
        depth_o3d = o3d.geometry.Image(depth_m)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=1.0, convert_rgb_to_intensity=False
        )
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsic_matrix
        )

        spheres = create_spheres(
            grasp_pred, contact_idx, separate_idx,
            radius=cfg["trajectory"]["sphere_radius"],
            highlight_radius=cfg["trajectory"]["highlight_radius"],
        )

        lines = create_thick_lines(
            grasp_pred,
            contact_idx,
            separate_idx,
            sphere_radius=cfg["trajectory"]["line_sphere_radius"],
            density=cfg["trajectory"]["line_density"],
        )

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.add_geometry(point_cloud)
        vis.add_geometry(spheres)
        vis.add_geometry(lines)

        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    main()