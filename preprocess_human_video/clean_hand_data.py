"""
Repository: https://github.com/IRMVLab/UniHand
Paper: Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
Authors: Ma et.al.

This file cleans 3D hand trajectories by interpolating zero-valued depth entries
and visualizes original vs refined joint trajectories.
"""


import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def update_zero_z(traj_data, filename):
    """
    Replace zero z-coordinates in the trajectory with interpolated values.
    traj_data: numpy array of shape (N, 21, 3)
    """
    N, num_joints, _ = traj_data.shape
    refined_traj_data = np.copy(traj_data)

    for joint_idx in range(num_joints):
        joint_data = traj_data[:, joint_idx, :]

        for t in range(N):
            if joint_data[t, 2] == 0:
                print(f"Zero z-coordinate found at joint {joint_idx}, time {t}: {joint_data[t]}")
                print("filename:", filename)

                # Search nearest non-zero z on the left
                left = t - 1
                while left >= 0 and joint_data[left, 2] == 0:
                    left -= 1

                # Search nearest non-zero z on the right
                right = t + 1
                while right < N and joint_data[right, 2] == 0:
                    right += 1

                # Interpolation / backward fill / forward fill
                if left >= 0 and right < N:
                    refined_traj_data[t, joint_idx, 2] = (joint_data[left, 2] + joint_data[right, 2]) / 2
                elif left >= 0:
                    refined_traj_data[t, joint_idx, 2] = joint_data[left, 2]
                elif right < N:
                    refined_traj_data[t, joint_idx, 2] = joint_data[right, 2]
                else:
                    # Fallback if all z are zero
                    if t > 0:
                        refined_traj_data[t, joint_idx, 2] = refined_traj_data[t - 1, joint_idx, 2]
                    elif t < N - 1:
                        refined_traj_data[t, joint_idx, 2] = refined_traj_data[t + 1, joint_idx, 2]
                    else:
                        refined_traj_data[t, joint_idx, 2] = 0
    return refined_traj_data



def visualize_trajectories(original_traj, refined_traj, joint_idx, save_path):
    """
    Visualize original vs refined trajectory for a given joint.
    The last trajectory point is highlighted with a larger marker.
    """
    original_joint = original_traj[:, joint_idx, :]
    refined_joint = refined_traj[:, joint_idx, :]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original trajectory
    ax.plot(original_joint[:, 0], original_joint[:, 1], original_joint[:, 2],
            color='r', label='Original Trajectory')

    # Plot refined trajectory
    ax.plot(refined_joint[:, 0], refined_joint[:, 1], refined_joint[:, 2],
            color='b', label='Refined Trajectory')

    # Highlight the last points
    ax.scatter(*original_joint[-1], s=100, color='r', label='Last Point (Original)')
    ax.scatter(*refined_joint[-1], s=100, color='b', label='Last Point (Refined)')

    ax.set_title(f'Trajectory Comparison (Joint {joint_idx})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Trajectory Cleaning + Visualization")
    parser.add_argument("--gt_paths", required=True, type=str,
                        help="Root folder containing all trajectory subfolders")
    parser.add_argument("--joint_idx", default=0, type=int,
                        help="Joint index to visualize (0-20)")

    args = parser.parse_args()

    gt_paths = args.gt_paths
    joint_to_visualize = args.joint_idx

    file_name_list = os.listdir(gt_paths)

    for filename in file_name_list:
        print("\n=================================")
        print("Processing:", filename)

        pkl_path = os.path.join(gt_paths, filename, "traj3d_in_cam_keypoints/traj3d_in_cam_dict.pkl")
        refine_path = os.path.join(gt_paths, filename, "traj3d_in_cam_keypoints/traj3d_in_cam_dict_refine.pkl")

        if not os.path.exists(pkl_path):
            print("Skipped: traj3d_in_cam_dict.pkl not found")
            continue

        # Load original data
        with open(pkl_path, "rb") as f:
            traj_raw_dict = pickle.load(f)

        traj_raw_list = [traj_raw_dict[key] for key in traj_raw_dict.keys()]
        traj_raw = np.array(traj_raw_list)

        # Clean zero-z
        refined_traj = update_zero_z(traj_raw, filename)

        # Save refined dictionary
        for key, new_frame in zip(traj_raw_dict.keys(), refined_traj):
            traj_raw_dict[key] = new_frame

        with open(refine_path, "wb") as f:
            pickle.dump(traj_raw_dict, f)

        print("Refined file saved:", refine_path)

        # Visualization
        viz_path = os.path.join(gt_paths, filename, f"viz_joint{joint_to_visualize}.png")
        visualize_trajectories(traj_raw, refined_traj, joint_to_visualize, viz_path)

    print("\nAll processing completed.")


if __name__ == "__main__":
    main()
