
"""
Repository: https://github.com/IRMVLab/UniHand
Paper: Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
Authors: Ma et.al.

This file contains trajectory processing utilities and the main training/evaluation loop 
for the Uni-Hand model's diffusion-based trajectory prediction.
"""

import time
import numpy as np
import torch
import os
from tqdm import trange
import functools
import logging.config
import datetime
from functools import partial
from denoising_diffusion.rounding import denoised_fn_round
from denoising_diffusion.step_sample import LossAwareSampler, UniformSampler
from denoising_diffusion.utils import dist_util, logger
from netscripts import modelio
from netscripts.epoch_utils import progress_bar as bar, AverageMeters
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.nn.functional as F
from einops import rearrange
from data_utils.utils import load_config


traineval_cfg = load_config("configs/traineval.yaml")
intrinsics = traineval_cfg['data']['intrinsics']

def gaussian_smooth(input_tensor, kernel_size=3, sigma=1.0):
    """
    Apply Gaussian smoothing to a 1D signal
    """
    x = torch.arange(kernel_size).float() - (kernel_size - 1) // 2
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1).to(input_tensor.device)
    padded = input_tensor.view(1, 1, -1)
    smoothed = F.conv1d(padded, kernel, padding=kernel_size//2)
    return smoothed.squeeze()

def find_local_peaks(signal, window_size=3):
    """
    Find local peaks in a 1D signal using max pooling
    """
    padded = F.pad(signal.unsqueeze(0), (window_size//2, window_size//2), mode='reflect')
    pooled = F.max_pool1d(padded, kernel_size=window_size, stride=1)
    peaks = (pooled == signal).float() * signal
    return peaks.squeeze(0)

def get_topk_peaks(signal, k=2, window_size=3):
    """
    Get top-k peaks from a signal
    """
    peaks = find_local_peaks(signal, window_size)
    peak_indices = torch.nonzero(peaks > 0).squeeze(-1)
    peak_values = peaks[peak_indices]
    if len(peak_values) >= k:
        topk_values, topk_idx = torch.topk(peak_values, k=k)
        topk_indices = peak_indices[topk_idx]
    else:
        topk_indices = F.pad(peak_indices, (0, k - len(peak_indices)), value=-1)
        topk_values = F.pad(peak_values, (0, k - len(peak_values)), value=0)
    
    return topk_indices, topk_values

def denormalize(traj, target='3d', max_depth=1.0):
    """
    Transform normalized (u,v,depth) coordinates to (X,Y,Z)
    """
    u = traj[:, 0]
    v = traj[:, 1]
    z = traj[:, 2] * max_depth
    traj3d = np.stack((u, v, z), axis=1)
    return traj3d

def denormalize_3d(traj, target='3d', max_depth=1.0):
    """
    Transform normalized (u,v,depth) coordinates to 3D world coordinates
    """
    u = traj[:, 0] * intrinsics['w']
    v = traj[:, 1] * intrinsics['h']

    Z = traj[:, 2] * max_depth
    X = (u - intrinsics['ox']) * Z / intrinsics['fx']
    Y = (v - intrinsics['oy']) * Z / intrinsics['fy']
    traj3d = np.stack((X, Y, Z), axis=1)
    return traj3d

def XYZ_to_uv(traj3d, intrinsics):
    """
    Convert 3D world coordinates to 2D pixel coordinates
    traj3d: (T, 3), a list of 3D (X, Y,Z) points 
    """
    traj2d = np.zeros((traj3d.shape[0], 2), dtype=np.float32)
    traj2d[:, 0] = (traj3d[:, 0] * intrinsics['fx'] / traj3d[:, 2] + intrinsics['ox'])
    traj2d[:, 1] = (traj3d[:, 1] * intrinsics['fy'] / traj3d[:, 2] + intrinsics['oy'])
    traj2d[:, 0] = np.clip(traj2d[:, 0], 0, intrinsics['w'])
    traj2d[:, 1] = np.clip(traj2d[:, 1], 0, intrinsics['h'])
    traj2d = np.floor(traj2d).astype(np.int32)
    return traj2d

def normalize_2d(uv, intrinsics):
    """
    Normalize 2D pixel coordinates to [0, 1] range
    """
    uv_norm = np.copy(uv).astype(np.float32)
    uv_norm[:, 0] /= intrinsics['w']
    uv_norm[:, 1] /= intrinsics['h']
    return uv_norm

def traj_affordance_dist(hand_traj, contact_point, future_valid=None, invalid_value=9):
    """
    Calculate distance between hand trajectory and contact point
    """
    batch_size = contact_point.shape[0]
    expand_size = int(hand_traj.shape[0] / batch_size)
    contact_point = contact_point.unsqueeze(dim=1).expand(-1, expand_size, 2).reshape(-1, 2)
    dist = torch.sum((hand_traj - contact_point) ** 2, dim=1).reshape(batch_size, -1)
    if future_valid is None:
        sorted_dist, sorted_idx = torch.sort(dist, dim=-1, descending=False)
        return sorted_dist[:, 0]
    else:
        dist = dist.reshape(batch_size, 2, -1)
        future_valid = future_valid > 0
        future_invalid = ~future_valid[:, :, None].expand(dist.shape)
        dist[future_invalid] = invalid_value
        sorted_dist, sorted_idx = torch.sort(dist, dim=-1, descending=False)
        selected_dist = sorted_dist[:, :, 0]
        selected_dist, selected_idx = selected_dist.min(dim=1)
        valid = torch.gather(future_valid, dim=1, index=selected_idx.unsqueeze(dim=1)).squeeze(dim=1)
        selected_dist = selected_dist * valid

    return selected_dist

def compute_fde(pred_traj, gt_traj, valid_traj=None, reduction=True):
    """
    Compute Final Displacement Error between predicted and ground truth trajectories
    """
    pred_last = pred_traj[:, :, -1, :]
    gt_last = gt_traj[:, :, -1, :]

    valid_loc = (gt_last[:, :, 0] >= 0) & (gt_last[:, :, 1] >= 0) \
                & (gt_last[:, :, 0] < 1) & (gt_last[:, :, 1] < 1)

    error = gt_last - pred_last
    error = error * valid_loc[:, :, None]

    if torch.is_tensor(error):
        if valid_traj is None:
            valid_traj = torch.ones(pred_traj.shape[0], pred_traj.shape[1])
        error = error ** 2
        fde = torch.sqrt(error.sum(dim=2)) * valid_traj
        if reduction:
            fde = fde.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()
    else:
        if valid_traj is None:
            valid_traj = np.ones((pred_traj.shape[0], pred_traj.shape[1]), dtype=int)
        error = np.linalg.norm(error, axis=2)
        fde = error * valid_traj
        if reduction:
            fde = fde.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()

    return fde, valid_traj

def compute_ade(pred_traj, gt_traj, valid_traj=None, reduction=True):
    """
    Compute Average Displacement Error between predicted and ground truth trajectories
    """
    valid_loc = (gt_traj[:, :, :, 0] >= 0) & (gt_traj[:, :, :, 1] >= 0)  \
                 & (gt_traj[:, :, :, 0] < 1) & (gt_traj[:, :, :, 1] < 1)

    error = gt_traj - pred_traj
    error = error * valid_loc[:, :, :, None]

    if torch.is_tensor(error):
        if valid_traj is None:
            valid_traj = torch.ones(pred_traj.shape[0], pred_traj.shape[1])
        error = error ** 2
        ade = torch.sqrt(error.sum(dim=3)).mean(dim=2) * valid_traj
        if reduction:
            ade = ade.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()
    else:
        if valid_traj is None:
            valid_traj = np.ones((pred_traj.shape[0], pred_traj.shape[1]), dtype=int)
        error = np.linalg.norm(error, axis=3)
        ade = (0.25*error[:,:,0]+0.5*error[:,:,1]+0.75*error[:,:,2]+1*error[:,:,3]) * valid_traj
        if reduction:
            ade = ade.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()

    return ade, valid_traj

def get_traj_observed(traj_all, num_ratios, mask_o):
    """
    Get the observed part of trajectory
    """
    traj_input = traj_all[:, None].repeat(1, num_ratios, 1, 1)
    traj_input = rearrange(traj_input, 'b n t c-> (b n) t c')
    traj_mask = mask_o.unsqueeze(-1)
    traj_input = traj_input * traj_mask
    return traj_input


def get_masks(batch_size, max_frames, nframes, device):
    """
    Generate observation and unobserved masks for trajectory
    """
    mask_o = torch.zeros((batch_size, 1, max_frames)).to(device, non_blocking=True)
    mask_u = torch.zeros((batch_size, 1, max_frames)).to(device, non_blocking=True)
    last_frames = torch.zeros((batch_size, 1)).long()
    mask_o[:, :, 0] = 1
    mask_u[:, :, 1:] = 1
    return mask_o, mask_u, last_frames


class TrainEvalLoop:
    def __init__(
            self,
            start_epoch=0,
            epochs=25,
            gap=1,
            max_frames=40,
            smooth_window=3,
            snapshot=500,
            loader=None,
            evaluate=False,
            use_cuda=True,
            scheduler=None,
            optimizer=None,
            pre_encoder=None,
            motion_encoder=None,
            loc_encoder=None,
            vision_encoder=None,
            voxel_encoder=None,
            occ_feat_encoder=None,
            model_denoise=None,
            homo_denoise=None,
            diffusion=None,
            homo_diffusion=None,
            post_encoder=None,
            contact_encoder=None,
            schedule_sampler=None,
            homo_schedule_sampler=None,
            resume=None,
            use_os_weights=True,
            save_pred=False,
            output_dir=None,
            loss_weights=None,
    ):
        """
        Initialize training/evaluation loop
        """

        self.unihand_models_dict = {
            'pre_encoder': pre_encoder,
            'model_denoise': model_denoise,
            'homo_denoise': homo_denoise,
            'post_encoder': post_encoder,
            'contact_encoder': contact_encoder,
            'motion_encoder': motion_encoder,
            'loc_encoder': loc_encoder,
            'vision_encoder': vision_encoder,
            'voxel_encoder': voxel_encoder,
            'occ_feat_encoder': occ_feat_encoder
        }

        # Initialize models and load checkpoints if available
        for attr_name, model in self.unihand_models_dict.items():
            setattr(self, attr_name, DDP(
                model.to(dist_util.dev()),
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            ))
            if os.path.exists(resume):
                state_dict_name = attr_name + "_state_dict"
                # TODO: update weight keys 
                if "vision" in state_dict_name and use_os_weights:
                    state_dict_name = state_dict_name.replace("vision", "glip")
                self.start_epoch = modelio.load_checkpoint_by_name(
                    getattr(self, attr_name), 
                    resume_path=resume, 
                    state_dict_name=state_dict_name, 
                    strict=False, 
                    device=dist_util.dev()
                )
            dist_util.sync_params(getattr(self, attr_name).parameters())


        self.diffusion = diffusion
        self.homo_diffusion = homo_diffusion
        self.start_epoch = start_epoch
        self.all_epochs = epochs
        self.evaluate = evaluate
        self.optimizer = optimizer
        self.loader = loader
        self.use_cuda = use_cuda
        self.scheduler = scheduler
        self.schedule_sampler = schedule_sampler
        self.homo_schedule_sampler = homo_schedule_sampler
        self.output_dir = output_dir
        self.loss_weights = loss_weights
        self.max_frames = max_frames
        self.infer_gap = gap
        self.smooth_window = smooth_window
        self.snapshot = snapshot

        self.ade_list, self.fde_list, self.pred_list, self.gt_list, self.ce_list = [], [], [], [], []

        if self.evaluate:
            self.all_epochs = 1
            self.start_epoch = 0

        self.logger = logging.getLogger('main')
        self.logger.setLevel(level=logging.DEBUG)
        now = datetime.datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        log_dir = "./log/"
        os.makedirs(log_dir, exist_ok=True)
        
        if evaluate:
            handler = logging.FileHandler(log_dir + f"eval_{time_str}.log")
        else:
            handler = logging.FileHandler(log_dir + f"{time_str}.log")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.save_pred = save_pred
        self.filename_all = []

    def _get_loc_features(self, traj, num_ratios, mask):
        """
        Extract location features from trajectory
        traj: (B, T, 3)
        num_ratios: ()
        mask: (B*N, T)
        """
        traj_input = get_traj_observed(traj, num_ratios, mask)
        loc_feat = self.loc_encoder(traj_input)
        return loc_feat

    def run_loop(self):
        """
        Main training/evaluation loop
        """
        # Set model modes
        for attr_name, model in self.unihand_models_dict.items():
            if not self.evaluate:
                getattr(self, attr_name).train()
            else:
                getattr(self, attr_name).eval()

        print("Num of epochs ", self.all_epochs)
        for epoch in range(self.start_epoch, self.all_epochs):
            if not self.evaluate:
                print("Using lr {}".format(self.optimizer.param_groups[0]["lr"]))
                self.logger.info("Using lr {}".format(self.optimizer.param_groups[0]["lr"]))
            self.epoch_pass(epoch=epoch)
            
    def epoch_pass(self, epoch):
        """
        Process one epoch of training or evaluation
        """
        time_meters = AverageMeters()

        if not self.evaluate:
            loss_meters = AverageMeters()
        else:
            print(f"evaluate epoch {epoch}")

        before_data_iter = time.time()
        for batch_idx, sample in enumerate(self.loader):

            if not self.evaluate:
                self.optimizer.zero_grad()

                # Extract data from batch
                traj3d_sampled, vision_feat_sampled, motion_feat_sampled, nframes, filename, valid_frame_index, vision_feat_first, voxel_feats_raw, contact_time = sample

                time_meters.add_loss_value("data_time", time.time() - before_data_iter)
                contact_time = contact_time.to(dist_util.dev())

                # Process voxel features
                voxel_feats_raw = voxel_feats_raw.unsqueeze(1).to(dist_util.dev())
                voxel_feats_all_space = self.voxel_encoder(voxel_feats_raw)
                voxel_feats_all_space = voxel_feats_all_space.squeeze(1)
                voxel_feats_all_space = voxel_feats_all_space.view(voxel_feats_all_space.shape[0], voxel_feats_all_space.shape[1], -1).contiguous()
                voxel_feats_all_space = voxel_feats_all_space.permute(0,2,1)
                occ_feat_encoded = self.occ_feat_encoder(voxel_feats_all_space.contiguous())

                # Process trajectory data
                traj3d_sampled = traj3d_sampled.to(dist_util.dev())
                traj_gt_wrist = traj3d_sampled[:, :, 0:1, :]
                traj_gt_3 = traj_gt_wrist  # TODO: add multi-finger

                motion_feats_raw = motion_feat_sampled
                vision_feat_first = vision_feat_first.to(dist_util.dev())  
                vision_feat_first_expanded = vision_feat_first.unsqueeze(1).expand(-1, self.max_frames, -1).contiguous()

                # Process each finger
                for finger_index in range(traj_gt_3.shape[2]):
                    traj_gt = traj_gt_3[:, :, finger_index, :]

                    batch_size, max_frames = traj_gt.size(0), traj_gt.size(1)
                    motion_feats_raw = motion_feats_raw.view(motion_feats_raw.shape[0], motion_feats_raw.shape[1], 3*3)
                    motion_feat_encoded = self.motion_encoder(motion_feats_raw)   
                    vision_feat_first_expanded = vision_feat_first_expanded.to(dist_util.dev())
                    vision_feat = self.vision_encoder(vision_feat_first_expanded)

                    self.output_size = 3
                    max_frames = vision_feat.shape[1]
                    traj_all = traj_gt[:, :, :self.output_size]
                    mask_o, mask_u, last_obs_frames = get_masks(batch_size, max_frames, nframes, dist_util.dev())
                    mask_o_for_input = rearrange(mask_o, 'b n t -> (b n) t')
                    mask_u_for_input = rearrange(mask_u, 'b n t -> (b n) t')

                    mask_ou_for_input = mask_o_for_input + mask_u_for_input
                    all_loc_feats = self._get_loc_features(traj_all, 1, mask_ou_for_input)

                    assert (torch.sum(mask_u_for_input[0]==1)+torch.sum(mask_o_for_input[0]==1)) == torch.sum(mask_ou_for_input[0]==1)

                    # Create finger indicator
                    finger_flag = torch.zeros(vision_feat.shape[0], vision_feat.shape[1], traj_gt_3.shape[2]).to(dist_util.dev())
                    finger_flag[:, :, finger_index] = 1.0

                    # Combine features
                    right_feat = torch.stack((vision_feat, all_loc_feats), dim=2)
                    right_feat = right_feat.view(*right_feat.shape[0:2], -1)
                    right_feat = torch.cat((right_feat, finger_flag), dim=2)

                    right_feat_encoded = self.pre_encoder(right_feat)

                    # Diffusion process
                    t, weights = self.schedule_sampler.sample(vision_feat.shape[0], dist_util.dev()) 

                    compute_losses = functools.partial(
                        self.diffusion.training_losses,
                        self.model_denoise,
                        self.post_encoder,
                        [right_feat_encoded, right_feat_encoded],
                        t,
                        [mask_ou_for_input, mask_o_for_input, mask_u_for_input],
                        [motion_feat_encoded,occ_feat_encoded],
                    )

                    loss_feat_dict = compute_losses()
                    loss_feat_level = loss_feat_dict["loss_feat_level"]
                    rec_feature_r = loss_feat_dict["rec_feature_r"]

                    future_feature = rec_feature_r
                    pred_future_traj = self.post_encoder(future_feature)
                    pred_future_traj_r = pred_future_traj

                    broaded_future_mask = torch.broadcast_to(mask_ou_for_input.unsqueeze(dim=-1), traj_gt.shape)

                    # Vanilla loss calculation
                    vanilla_loss_weight = self.loss_weights[0]
                    vanilla_future_traj_r = self.post_encoder(right_feat_encoded.contiguous())
                    vanilla_future_traj_r_masked = vanilla_future_traj_r * broaded_future_mask
                    traj_gt_masked = traj_gt * broaded_future_mask
                    vanilla_r_loss = torch.sum((vanilla_future_traj_r_masked - traj_gt_masked) ** 2, dim=-1)
                    vanilla_r_loss = vanilla_r_loss.sum(-1) * vanilla_loss_weight

                    broaded_future_mask_for_angle = mask_u_for_input

                    # Angle loss calculation
                    pred_future_traj_r_shifted_back = pred_future_traj_r[:, 1:, :]
                    pred_future_traj_r_shifted_forward = pred_future_traj_r[:, 0:-1, :]
                    pred_future_traj_r_shifted_delta = pred_future_traj_r_shifted_back - pred_future_traj_r_shifted_forward

                    future_head_r =  traj_gt
                    future_head_r_shifted_back = future_head_r[:, 1:, :]
                    future_head_r_shifted_forward = future_head_r[:, 0:-1, :]
                    future_head_r_shifted_delta = future_head_r_shifted_back - future_head_r_shifted_forward

                    cos_sim = F.cosine_similarity(pred_future_traj_r_shifted_delta, future_head_r_shifted_delta, dim=-1)
                    cos_distance = 1 - cos_sim

                    cos_distance = torch.cat((cos_distance, torch.zeros((pred_future_traj_r.shape[0],1)).to(dist_util.dev())),dim=-1)
                    cos_distance = cos_distance * broaded_future_mask_for_angle

                    angle_loss_weight = self.loss_weights[1]
                    future_angle_r_loss = torch.sum(cos_distance, dim=-1)*angle_loss_weight

                    # Trajectory loss calculation
                    pred_future_traj_r_masked = pred_future_traj_r * broaded_future_mask
                    traj_gt_masked = traj_gt * broaded_future_mask
                    future_traj_r_loss = torch.sum((pred_future_traj_r_masked - traj_gt_masked) ** 2, dim=-1)
                    future_traj_r_loss = future_traj_r_loss.sum(-1)

                    rec_loss_weight = self.loss_weights[2]

                    losses_r = rec_loss_weight * loss_feat_level['mse_r'] + loss_feat_level['tT_loss_r'] + future_traj_r_loss + vanilla_r_loss + future_angle_r_loss

                    if finger_index == 0:
                        losses_two_hand = losses_r
                    else:
                        losses_two_hand = losses_two_hand + losses_r

                # Contact time prediction
                pred_contact_time = self.contact_encoder(future_feature)
                pred_contact_time = pred_contact_time[:,:,0]
                broaded_future_mask_contact = broaded_future_mask[:,:,0]
                criterion_contact = torch.nn.BCELoss()           
                gt_contact_time = torch.nn.functional.one_hot(contact_time, num_classes=self.max_frames).to(dist_util.dev())
                gt_contact_time = gt_contact_time.sum(dim=1)
                gt_contact_time = gt_contact_time.float()
                pred_contact_time = pred_contact_time * broaded_future_mask_contact
                gt_contact_time = gt_contact_time * broaded_future_mask_contact
                loss_contact = criterion_contact(pred_contact_time, gt_contact_time)

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses_two_hand.detach()
                    )

                loss = (losses_two_hand * weights).mean() 
                loss = loss + loss_contact

                model_losses = {
                        "future_traj_r_loss":future_traj_r_loss.mean(),
                        "future_angle_r_loss":future_angle_r_loss.mean(),
                        "rec_r_loss":loss_feat_level['mse_r'].mean(),
                        "loss_contact": loss_contact,
                        "total_loss":loss,
                }

                loss.backward()
                self.optimizer.step()

                for key, val in model_losses.items():
                    if val is not None:
                        loss_meters.add_loss_value(key, val.detach().cpu().item())

                time_meters.add_loss_value("batch_time", time.time() - before_data_iter)

                if dist_util.get_rank() == 0:
                    self.logger.info(loss_meters.average_meters["total_loss"].avg)

                    suffix = "Epoch:{epoch} " \
                            "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s " \
                            "| future_traj_r_loss: {future_traj_r_loss:.3f} " \
                            "| future_angle_r_loss: {future_angle_r_loss:.3f} " \
                            "| rec_r_loss: {rec_r_loss:.3f}" \
                            "| loss_contact: {loss_contact:.3f}" \
                            "| total_loss: {total_loss:.3f} ".format(epoch=epoch, batch=batch_idx + 1, size=len(self.loader),
                                                                    data=time_meters.average_meters["data_time"].val,
                                                                    bt=time_meters.average_meters["batch_time"].avg,
                                                                    future_traj_r_loss=model_losses["future_traj_r_loss"],
                                                                    future_angle_r_loss=model_losses["future_angle_r_loss"],
                                                                    rec_r_loss=model_losses["rec_r_loss"],
                                                                    loss_contact=model_losses["loss_contact"],
                                                                    total_loss=model_losses["total_loss"],
                                                                    )
                    self.logger.info(suffix)
                    bar(suffix)

                end = time.time()

            else:  # Evaluation mode
                traj3d_sampled, vision_feat_sampled, motion_feat_sampled, nframes, filename, valid_frame_index, vision_feat_first, voxel_feats_raw, contact_time = sample

                voxel_feats_raw = voxel_feats_raw.to(dist_util.dev())
                voxel_feats_raw = voxel_feats_raw.unsqueeze(1)
                voxel_feats_all_space = self.voxel_encoder(voxel_feats_raw)
                voxel_feats_all_space = voxel_feats_all_space.squeeze(1)
                voxel_feats_all_space = voxel_feats_all_space.view(voxel_feats_all_space.shape[0], voxel_feats_all_space.shape[1], -1).contiguous()
                voxel_feats_all_space = voxel_feats_all_space.permute(0,2,1)
                occ_feat_encoded = self.occ_feat_encoder(voxel_feats_all_space.contiguous())

                traj3d_sampled = traj3d_sampled.to(dist_util.dev())
                traj_gt_wrist = traj3d_sampled[:, :, 0:1, :]
                traj_gt_3 = traj_gt_wrist # TODO: add multi-finger

                motion_feats_raw = motion_feat_sampled
                vision_feat_first = vision_feat_first.to(dist_util.dev())  
                vision_feat_first_expanded = vision_feat_first.unsqueeze(1).expand(-1, self.max_frames, -1).contiguous()

                # Process each finger
                for finger_index in range(traj_gt_3.shape[2]):
                    traj_gt = traj_gt_3[:, :, finger_index, :]

                    batch_size, max_frames = traj_gt.size(0), traj_gt.size(1)

                    motion_feats_raw = motion_feats_raw.view(motion_feats_raw.shape[0], motion_feats_raw.shape[1], 3*3)
                    motion_feat_encoded = self.motion_encoder(motion_feats_raw)   

                    vision_feat = self.vision_encoder(vision_feat_first_expanded)

                    finger_flag = torch.zeros(vision_feat.shape[0], vision_feat.shape[1], traj_gt_3.shape[2]).to(dist_util.dev())
                    finger_flag[:, :, finger_index] = 1.0
        
                    self.output_size = 3
                    max_frames = vision_feat.shape[1]
                    traj_all = traj_gt[:, :, :self.output_size]
                    mask_o, mask_u, last_obs_frames = get_masks(batch_size, max_frames, nframes, dist_util.dev())
                    mask_o_for_input = rearrange(mask_o, 'b n t -> (b n) t')
                    mask_u_for_input = rearrange(mask_u, 'b n t -> (b n) t')

                    mask_ou_for_input = mask_o_for_input + mask_u_for_input
                    all_loc_feats = self._get_loc_features(traj_all, 1, mask_ou_for_input)

                    assert (torch.sum(mask_u_for_input[0]==1)+torch.sum(mask_o_for_input[0]==1)) == torch.sum(mask_ou_for_input[0]==1)

                    right_feat = torch.stack((vision_feat, all_loc_feats), dim=2)
                    right_feat = right_feat.view(*right_feat.shape[0:2], -1)
                    right_feat = torch.cat((right_feat, finger_flag), dim=2)

                    right_feat_encoded = self.pre_encoder(right_feat)                

                    sample_fn = (
                        self.diffusion.p_sample_loop
                    )

                    with torch.no_grad():
                        len_observation = last_obs_frames + 1
                        assert len_observation[0] == int(torch.sum(mask_o_for_input[0]==1).item())

                        # Sample multiple times
                        for sample_idx in range(1):  # TODO: Multi-sample
                            for lo in range(len_observation.shape[0]):
                                nframes_this_b = int(nframes[lo])
                                nfuture = int(nframes_this_b - len_observation[lo])
                                pseudo_future = torch.zeros((nfuture,1024))
                                noise_r = torch.randn_like(pseudo_future).to(dist_util.dev())
                                right_feat_encoded[lo, len_observation[lo]:nframes_this_b, :] = noise_r

                            sample_shape = (right_feat_encoded.shape[0], right_feat_encoded.shape[1], right_feat_encoded.shape[2])

                            model_kwargs = {}
                            print("denoising ...")

                            start_idx = 0
                            samples_r = sample_fn(
                                model_denoise=self.model_denoise,
                                shape=sample_shape,
                                noise=[right_feat_encoded[:,start_idx:,...], right_feat_encoded[:,start_idx:,...]],
                                motion_feat_encoded=[motion_feat_encoded[:,start_idx:,...],occ_feat_encoded[:,start_idx:,...]],
                                clip_denoised=False,
                                model_kwargs=model_kwargs,
                                clamp_step=0,
                                clamp_first=True,
                                x_start=[right_feat_encoded, right_feat_encoded],
                                gap=self.infer_gap,
                                device=dist_util.dev(),
                                valid_mask = [mask_ou_for_input, mask_o_for_input, mask_u_for_input],
                            )

                            samples_r = samples_r[-1]
                            pred_future_traj = self.post_encoder(samples_r)
                            pred_contact_point = self.contact_encoder(samples_r)

                            # Smooth predictions
                            kernel_size = self.smooth_window
                            kernel = torch.ones(1, 1, kernel_size, dtype=torch.float32, device="cuda") / kernel_size
                            smoothed = []
                            for ixyz in range(3):
                                channel_data = pred_future_traj[:, :, ixyz].unsqueeze(1)
                                smoothed_channel = F.conv1d(channel_data, kernel, padding=kernel_size // 2)
                                smoothed_channel = smoothed_channel.squeeze(1)
                                smoothed_channel[:, 0] = pred_future_traj[:, 0, ixyz]
                                smoothed_channel[:, -1] = pred_future_traj[:, -1, ixyz]
                                smoothed.append(smoothed_channel)
                            pred_future_traj = torch.stack(smoothed, dim=-1)

                            # Evaluate predictions
                            for b in range(samples_r.shape[0]):
                                num_full = nframes[b]
                                traj_gt_per = traj_gt[b]
                                traj_gt_per += 0.5
                                traj_gt_per = denormalize_3d(traj_gt_per.cpu().numpy(), target='3d')
                                
                                samples_r_per = pred_future_traj[b]
                                samples_r_per += 0.5
                                samples_r_per = denormalize_3d(samples_r_per.cpu().numpy(), target='3d') 

                                start_ = int(len_observation[b])
                                end_ = int(num_full)

                                traj_gt_per_valid = traj_gt_per[start_:end_]
                                samples_r_per_valid = samples_r_per[start_:end_]

                                self.filename_all.append(filename)

                                valid_frame_index_per_valid = valid_frame_index[b, start_:end_]

                                displace_errors = np.sqrt(np.sum((samples_r_per_valid - traj_gt_per_valid)**2, axis=-1))
                                
                                ade = np.mean(displace_errors)
                                self.ade_list.append(ade)

                                final_displace_errors = np.sqrt(np.sum((samples_r_per_valid[-1] - traj_gt_per_valid[-1])**2, axis=-1))
                                fde = final_displace_errors
                                self.fde_list.append(fde)

                                pred_contact_point_per = pred_contact_point[b]
                                pred_contact_point_per_valid = pred_contact_point_per[start_:end_]
                                smoothed = gaussian_smooth(pred_contact_point_per_valid)
                                contact_indices, peak_values = get_topk_peaks(smoothed, k=2)
                                contact_point_gt_per = contact_time[b]
                                sorted_contact_indices, _ = torch.sort(contact_indices)
                                contact_shift = torch.mean(torch.abs(contact_point_gt_per.cpu()-sorted_contact_indices.cpu()).float())
                                self.ce_list.append(contact_shift.numpy())

                            print(str(batch_idx) + "/" + str(len(self.loader)) + " ADE " + str(np.array(self.ade_list).mean()) + " FDE " + str(np.array(self.fde_list).mean()) + " CE " + str(np.array(self.ce_list).mean()))
                            self.logger.info(str(batch_idx) + " unihand "+str(np.array(self.ade_list).mean())+" fde "+ str(np.array(self.fde_list).mean()))

                            assert len(filename) == 1
                            assert pred_future_traj.shape[0] == 1 
                            name_cur = filename[0].split('.')[0]
                            name_cur = name_cur.replace('/', '__')

                            if finger_index == 0:
                                finger_dir = "finger0"
                            elif finger_index == 1:
                                finger_dir = "finger4_8"

                            if self.save_pred:
                                save_data_dict = {
                                    'seed1': samples_r_per_valid,
                                    'saved_gt_all': traj_gt_per[:len_observation[0]],
                                    'saved_gt_future': traj_gt_per_valid,
                                    'start_index': start_,
                                    'end_index': end_,
                                    'valid_frame_index': valid_frame_index_per_valid,
                                    'contact_time_pred': sorted_contact_indices.cpu().numpy(),
                                    'contact_time_gt': contact_point_gt_per.cpu().numpy()
                                }

                                # Save prediction results
                                for folder_name, data in save_data_dict.items():
                                    save_path = os.path.join(self.output_dir, finger_dir, folder_name)
                                    os.makedirs(save_path, exist_ok=True)
                                    np.save(os.path.join(save_path, name_cur), data)
                  
        # Save checkpoints during training
        if not self.evaluate:
            warmup_epochs = 0
            if (epoch + 1 - warmup_epochs) % self.snapshot == 0 and (dist_util.get_rank() == 0):
                print("save epoch "+str(epoch+1)+" checkpoint")
                
                checkpoint_dir = "./diffip_weights"
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                modelio.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "pre_encoder_state_dict": self.pre_encoder.state_dict(),
                    "model_denoise_state_dict": self.model_denoise.state_dict(),
                    "homo_denoise_state_dict": self.homo_denoise.state_dict(),
                    "post_encoder_state_dict": self.post_encoder.state_dict(),
                    "contact_encoder_state_dict": self.contact_encoder.state_dict(),
                    "motion_encoder_state_dict": self.motion_encoder.state_dict(),
                    "loc_encoder_state_dict": self.loc_encoder.state_dict(),
                    "vision_encoder_state_dict": self.vision_encoder.state_dict(),
                    "voxel_encoder_state_dict": self.voxel_encoder.state_dict(),
                    "occ_feat_encoder_state_dict": self.occ_feat_encoder.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                checkpoint=checkpoint_dir,
                filename = f"checkpoint_{epoch+1}.pth.tar")
                torch.save(self.optimizer.state_dict(), "optimizer.pt")

                return loss_meters
        else:  # Return evaluation metrics
            val_info = {}
            val_info["ADE"] = np.array(self.ade_list).mean()
            val_info["FDE"] = np.array(self.fde_list).mean()
            val_info["CE"] = np.array(self.ce_list).mean()

            self.logger.info("ADE "+str(val_info["ADE"]))
            self.logger.info("FDE "+str(val_info["FDE"]))
            self.logger.info("CE "+str(val_info["CE"]))

            return val_info