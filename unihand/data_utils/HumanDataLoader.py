"""
Repository: https://github.com/IRMVLab/UniHand
Paper: Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
Authors: Ma et.al.

This file contains the dataset loader for hand trajectory prediction,
including data loading, preprocessing and augmentation for multimodal inputs.
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import pickle
import numpy as np
from functools import reduce
import open3d as o3d
import csv


class HumanData3D(Dataset):
    def __init__(self, phase='train', transform=None, data_cfg=None):
        """
        Initialize HumanData3D
        
        Args:
            phase (str): Dataset phase ('train', 'val', 'test', 'test_novel')
            transform: Data transformation function
            data_cfg (dict): Data configuration parameters
        """
        self.centralize = True
        self.data_cfg = data_cfg
        self.intrinsics = data_cfg['intrinsics'] 
        self.max_depth = data_cfg['max_depth']
        self.max_frames = data_cfg['max_frames']
        train_splits_file = data_cfg['train_splits_file']
        test_splits_file = data_cfg['test_splits_file']
        contact_label_file = data_cfg['contact_label_file']
        self.date_dir = data_cfg["data_root"]
        self.date_path_all = sorted([p for p in os.listdir(self.date_dir)])
        self.vision_feats_dir = data_cfg["feats_path"]
        self.traj_dir = data_cfg["traj_path"]
        self.interval = data_cfg["interval"]

        # Initialize data containers
        self.nframes = []
        self.traj_all = []
        self.filenames = []
        self.vision_feats_all = []
        self.valid_frame_name_all = []
        self.vision_feats_of_first_frame_all = []
        self.contact_time = []
        self.split_filenames = []

        # Load split filenames based on phase
        if phase == 'train':
            with open(train_splits_file, "r") as train_file:
                self.split_filenames = [line.strip() for line in train_file]
        else:
            with open(test_splits_file, "r") as test_file:
                self.split_filenames = [line.strip() for line in test_file]
        
        # Load contact time annotations
        date_contact_time_dict = {}
        with open(contact_label_file, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                date_key = row[0].strip()
                time_value_list = []
                for contact_long_idx in range(1, 3):
                    time_value_list.append(int(row[contact_long_idx].strip()))
                date_contact_time_dict[date_key] = time_value_list

        # Process each data sample
        for date in self.date_path_all:
            if not (date in self.split_filenames):
                continue
            
            self.filenames.append(date)
            
            # Load trajectory data
            traj_filename = os.path.join(self.traj_dir, date, "traj3d_in_cam_keypoints", "traj3d_in_cam_dict_refine.pkl")
            with open(traj_filename, "rb") as f:
                traj_data_raw_dict = pickle.load(f)

            # Load contact time information
            contact_time_current_state = date_contact_time_dict[date]
            contact_time_current_state_arr = np.array(contact_time_current_state).astype(int)
            self.contact_time.append(contact_time_current_state_arr)

            # Load visual features
            vision_feats_filename_path = os.path.join(self.vision_feats_dir, date + ".npy")
            vision_feats_data = np.load(vision_feats_filename_path)
            self.vision_feats_of_first_frame_all.append(vision_feats_data[0])

            # Match frames with valid trajectory data
            raw_data_path = os.path.join(self.date_dir, date, "rgb")
            raw_file_names = os.listdir(raw_data_path)
            sorted_raw_file_names = sorted(raw_file_names)
            
            traj_points = []
            vision_features = []
            valid_frame_names = []
            file_cnt = -1
            
            for raw_file in sorted_raw_file_names:
                file_cnt += 1
                if file_cnt % self.interval != 0:
                    continue
                    
                index_key = int(raw_file.split('.')[0])
                if index_key in traj_data_raw_dict.keys():
                    # Store valid trajectory point and corresponding features
                    traj_points.append(traj_data_raw_dict[index_key])
                    vision_features.append(vision_feats_data[int(file_cnt/self.interval)])
                    valid_frame_names.append(index_key)

            # Convert to numpy arrays
            traj_points = np.array(traj_points)
            vision_features = np.array(vision_features)
            valid_frame_names = np.array(valid_frame_names)

            # Process trajectory data (normalize and pad)
            processed_traj = np.zeros_like(traj_points)
            processed_traj[:, :, 0:2] = traj_points[:, :, 0:2]
            processed_traj[:, :, 2] = traj_points[:, :, 2] / 1000.0  # Convert from mm to meters
            processed_traj = self._normalize(processed_traj)
            
            # Pad or truncate to max_frames
            traj_data_padded = self.adjust_array(processed_traj)
            self.traj_all.append(traj_data_padded)
            self.nframes.append(min(traj_points.shape[0], self.max_frames))

            # Process visual features (pad or truncate)
            vision_features_padded = self.adjust_array(vision_features)
            self.vision_feats_all.append(vision_features_padded)

            # Process valid frame names
            valid_frame_names = valid_frame_names.reshape(-1, 1)
            valid_frame_names_padded = self.adjust_array(valid_frame_names)
            self.valid_frame_name_all.append(valid_frame_names_padded)

            assert vision_features_padded.shape[0] == traj_data_padded.shape[0]

        assert len(self.vision_feats_all) == len(self.traj_all)

    def adjust_array(self, source_data):
        """
        Pad or truncate array to match max_frames
        
        Args:
            source_data (np.array): Input data array
            
        Returns:
            np.array: Padded or truncated array
        """
        n = source_data.shape[0]
        if n < self.max_frames:
            # Pad with zeros to reach max_frames
            padding = np.zeros((self.max_frames - n, *source_data.shape[1:]))
            padded_data = np.vstack((source_data, padding))
        elif n > self.max_frames:
            # Truncate to max_frames
            padded_data = source_data[:self.max_frames]
        else:
            padded_data = source_data
        return padded_data

    def _XYZ_to_uv(self, traj3d):
        """
        Convert 3D coordinates to 2D image coordinates using pinhole camera model
        
        Args:
            traj3d (torch.Tensor): 3D trajectory points (N, 3)
            
        Returns:
            torch.Tensor: 2D image coordinates (N, 2)
        """
        width = self.intrinsics['w']
        height = self.intrinsics['h']
        
        # Apply pinhole camera model
        u = (traj3d[:, 0] * self.intrinsics['fx'] / traj3d[:, 2] + self.intrinsics['ox'])
        v = (traj3d[:, 1] * self.intrinsics['fy'] / traj3d[:, 2] + self.intrinsics['oy'])
        
        # Clamp coordinates to image boundaries
        u = torch.clamp(u, min=0, max=width-1)
        v = torch.clamp(v, min=0, max=height-1)
        
        traj2d = torch.stack((u, v), dim=-1)
        return traj2d

    def _normalize(self, traj):
        """
        Normalize trajectory coordinates to [0, 1] range
        
        Args:
            traj (np.array): Trajectory data (N, 21, 3)
            
        Returns:
            np.array: Normalized trajectory data
        """
        traj_new = traj.copy()
        width = self.intrinsics['w'] 
        height = self.intrinsics['h']
        
        # Normalize coordinates
        traj_new[:, :, 0] = traj_new[:, :, 0] / width
        traj_new[:, :, 1] = traj_new[:, :, 1] / height
        traj_new[:, :, 2] = traj_new[:, :, 2] / self.max_depth
        traj_new -= 0.5  # Center around 0
        return traj_new

    def _get_projected_traj3d(self, traj3d, odometry):
        """
        Project 3D trajectory points using odometry information
        
        Args:
            traj3d (np.array): 3D trajectory points (T, 3)
            odometry (np.array): Odometry data (T, 4, 4)
            
        Returns:
            np.array: Projected trajectory points
        """
        length = len(odometry)
        traj3d_homo = np.hstack((traj3d, np.ones((length, 1))))  # Convert to homogeneous coordinates
        all_traj3d_proj = []
        
        for i in range(length):
            # Compute transformed points for all future points
            traj3d_proj = [traj3d[i]]  # Initial 3D point
            for j in range(i + 1, length):
                odom = reduce(np.dot, odometry[(i+1):(j+1)]) 
                future_point = odom.dot(traj3d_homo[j].T) 
                traj3d_proj.append(future_point[:3])
            all_traj3d_proj.append(np.array(traj3d_proj))
            
        all_traj3d_proj = np.concatenate(all_traj3d_proj, axis=0) 
        return all_traj3d_proj
    
    def __len__(self):
        return len(self.traj_all)
    
    def __getitem__(self, index):
        index_for_feat = int(index)

        # Load trajectory data
        traj3d = torch.from_numpy(self.traj_all[index]).to(torch.float32)

        # Load visual features
        vision_feat = torch.from_numpy(self.vision_feats_all[index_for_feat]).to(torch.float32)
        vision_feat_first = torch.from_numpy(self.vision_feats_of_first_frame_all[index_for_feat]).to(torch.float32)

        # Sample data (no downsampling in this case)
        self.sample_interval = 1
        traj3d_sampled = traj3d[::self.sample_interval]
        vision_feat_sampled = vision_feat[::self.sample_interval]
        motion_feat_sampled = torch.zeros((vision_feat_sampled.shape[0], 3, 3))[::self.sample_interval]
        
        nframes = self.nframes[index]
        filename = self.filenames[index]

        # Load and process point cloud data to voxel grid
        # We don't load all point clouds in self.__init__() to reduce memory usage
        res = self.data_cfg['voxel_resolution']
        grid_size = self.data_cfg['grid_size']
        origin_xyz = self.data_cfg['origin_xyz']
        voxel_grid = torch.zeros((grid_size, grid_size, grid_size, 1))
        
        point_cloud_path = os.path.join(self.date_dir, filename, filename + "_point_cloud.ply")
        point_cloud = o3d.io.read_point_cloud(point_cloud_path)
        assert np.asarray(point_cloud.points).shape[0] != 0
        
        transformed_points_with_colors = np.asarray(point_cloud.points)
        coord_origin_point = np.tile(np.array(origin_xyz), (transformed_points_with_colors.shape[0], 1))
        transformed_points_with_colors[:, :3] = (transformed_points_with_colors[:, :3] - coord_origin_point) / res
        
        # Filter points within grid boundaries
        kept = np.all((transformed_points_with_colors[:, :3] >= 0) & 
                     (transformed_points_with_colors[:, :3] < grid_size-1), axis=1)
        transformed_points_with_colors = transformed_points_with_colors[kept]
        
        # Create voxel grid
        ones_column = torch.ones((transformed_points_with_colors.shape[0], 1))
        transformed_points_with_colors = torch.from_numpy(transformed_points_with_colors)
        voxel_grid[torch.round(transformed_points_with_colors[:, 0]).long(), 
                  torch.round(transformed_points_with_colors[:, 1]).long(),
                  torch.round(transformed_points_with_colors[:, 2]).long()] = ones_column.to(torch.float32)

        # Load valid frame indices
        valid_frame_index = torch.from_numpy(self.valid_frame_name_all[index]).to(torch.float32)

        # Process contact time information
        contact_time_arr = self.contact_time[index]
        target_time_list = []
        
        for contact_time_arr_idx in range(contact_time_arr.shape[0]):
            target_time = int(contact_time_arr[contact_time_arr_idx])
            contact_time_in_valid = (valid_frame_index == target_time).nonzero(as_tuple=True)[0]
            
            if contact_time_in_valid.numel() <= 0:
                print("Warning: Hand detection or annotation mismatch!")
                print("Error details:")
                print("filename", filename)  
                print("target_time", target_time)
                print("valid_frame_index", valid_frame_index)
                print("contact_time_in_valid", contact_time_in_valid)
            else:
                target_time_list.append(contact_time_in_valid.item())  

        contact_time_in_valid_arr = np.array(target_time_list)

        return (traj3d_sampled, vision_feat_sampled, motion_feat_sampled, nframes, 
                filename, valid_frame_index, vision_feat_first, voxel_grid, contact_time_in_valid_arr)


def build_dataloaders(traineval_cfg, data_cfg, phase='trainval'):
    """
    Build data loaders for training/validation or testing
    
    Args:
        traineval_cfg (dict): Training/evaluation configuration
        data_cfg (dict): Data configuration
        phase (str): Phase ('trainval' or other)
        
    Returns:
        tuple: Data loaders
    """
    if phase == 'trainval':
        # Create training set
        trainset = HumanData3D(phase='train', transform=None, data_cfg=data_cfg)
        train_loader = DataLoader(
            trainset, 
            batch_size=traineval_cfg["batch_size"], 
            shuffle=True, 
            num_workers=data_cfg["num_workers"], 
            pin_memory=True
        )
        
        # Create validation set
        valset = HumanData3D(phase='val', transform=None, data_cfg=data_cfg)
        val_loader = DataLoader(
            valset, 
            batch_size=traineval_cfg["batch_size"], 
            shuffle=False, 
            num_workers=data_cfg["num_workers"], 
            pin_memory=True
        )
        
        print("Number of train/val samples: {}/{}".format(len(trainset), len(valset)))
        return train_loader, val_loader
    else:
        # Create test sets
        testset = HumanData3D(phase='test', transform=None, data_cfg=data_cfg)
        test_loader = DataLoader(
            testset, 
            batch_size=traineval_cfg["batch_size"], 
            shuffle=False, 
            num_workers=data_cfg["num_workers"], 
            pin_memory=True
        )
        
        testnovel_set = HumanData3D(phase='test_novel', transform=None, data_cfg=data_cfg)
        testnovel_loader = DataLoader(
            testnovel_set, 
            batch_size=traineval_cfg["batch_size"], 
            shuffle=False, 
            num_workers=data_cfg["num_workers"], 
            pin_memory=True
        )
        
        print("Number of test/test_novel samples: {}/{}".format(len(testset), len(testnovel_set)))
        return test_loader, testnovel_loader