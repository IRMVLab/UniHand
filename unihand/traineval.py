"""
Repository: https://github.com/IRMVLab/UniHand
Paper: Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
Authors: Ma et.al.

This file contains the main training and evaluation pipeline for the Uni-Hand model,
which predicts hand trajectories using multimodal diffusion models.

Key Components:
1. Hand Trajectory Prediction (HTP) diffusion model
2. Egomotion diffusion model (temporarily unavailable for static robot base)
3. Various encoders and decoders for processing different input modalities
4. Training/evaluation loop with GPU support
"""

import os
import random
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy("file_system")

from data_utils.utils import load_config

from netscripts.get_optimizer import get_optimizer
from netscripts.epoch_feat import TrainEvalLoop
from basic_utils import create_model_and_diffusion, homo_create_model_and_diffusion
from denoising_diffusion.step_sample import create_named_schedule_sampler
from denoising_diffusion.utils import dist_util

from data_utils.HumanDataLoader import build_dataloaders


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(model_cfg, train_cfg, eval_cfg, data_cfg, common_cfg):

    # ----------------------------
    # Setup
    # ----------------------------
    set_random_seed(train_cfg["manual_seed"])
    dist_util.setup_dist()

    # ----------------------------
    # Build diffusion models
    # ----------------------------
    diff_args = model_cfg["diffusion"]
    (
        pre_encoder,
        model_denoise,
        diffusion,
        post_encoder,
        motion_encoder,
        loc_encoder,
        vision_encoder,
        voxel_encoder,
        occ_feat_encoder,
        contact_encoder,
    ) = create_model_and_diffusion(model_cfg, **diff_args)

    homo_denoise, homo_diffusion = homo_create_model_and_diffusion(model_cfg, **diff_args)
    print("Diffusion models built.")

    # Samplers
    sampler_name = model_cfg["schedule_sampler"]
    schedule_sampler = create_named_schedule_sampler(sampler_name, diffusion)
    homo_schedule_sampler = create_named_schedule_sampler(sampler_name, homo_diffusion)

    # ----------------------------
    # Build datasets
    # ----------------------------

    print("common_cfg[evaluate]", common_cfg["evaluate"])

    if common_cfg["evaluate"]:
        train_loader = None
        test_loader, testnovel_loader = build_dataloaders(eval_cfg, data_cfg, phase="test")
        loader = test_loader if eval_cfg["unseen_envs"] else testnovel_loader
        optimizer = scheduler = None
    else:
        train_loader, val_loader = build_dataloaders(train_cfg, data_cfg, phase="trainval")
        loader = train_loader

        optimizer, scheduler = get_optimizer(
            train_cfg,
            pre_encoder=pre_encoder,
            model_denoise=model_denoise,
            homo_denoise=homo_denoise,
            post_encoder=post_encoder,
            contact_encoder=contact_encoder,
            loc_encoder=loc_encoder,
            vision_encoder=vision_encoder,
            voxel_encoder=voxel_encoder,
            occ_feat_encoder=occ_feat_encoder,
            train_loader=train_loader,
            motion_encoder=motion_encoder,
        )

    # ----------------------------
    # Create output dir
    # ----------------------------
    output_dir = common_cfg["save"]["output_dir"]
    ensure_dir(output_dir)

    # ----------------------------
    # Training loop
    # ----------------------------

    TrainEvalLoop(
        epochs=train_cfg["epochs"],
        resume=common_cfg["resume"],
        loss_weights=train_cfg["loss_weights"],
        snapshot=train_cfg["snapshot"],
        gap=eval_cfg["gap"],
        smooth_window=eval_cfg["smooth_window"],
        use_os_weights=eval_cfg["use_os_weights"],
        max_frames=data_cfg["max_frames"],
        save_pred=common_cfg["save"]["save_traj"],
        evaluate=common_cfg["evaluate"],
        use_cuda=common_cfg["use_cuda"],
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        pre_encoder=pre_encoder,
        model_denoise=model_denoise,
        homo_denoise=homo_denoise,
        diffusion=diffusion,
        homo_diffusion=homo_diffusion,
        post_encoder=post_encoder,
        contact_encoder=contact_encoder,
        motion_encoder=motion_encoder,
        loc_encoder=loc_encoder,
        vision_encoder=vision_encoder,
        voxel_encoder=voxel_encoder,
        occ_feat_encoder=occ_feat_encoder,
        schedule_sampler=schedule_sampler,
        homo_schedule_sampler=homo_schedule_sampler,
        output_dir=output_dir,
    ).run_loop()


if __name__ == "__main__":


    model_cfg = load_config("configs/model.yaml")
    traineval_cfg = load_config("configs/traineval.yaml")

    train_cfg = traineval_cfg["train"]
    eval_cfg = traineval_cfg["eval"]
    data_cfg = traineval_cfg["data"]
    common_cfg = traineval_cfg


    main(model_cfg,train_cfg, eval_cfg, data_cfg, common_cfg)
    print("All done!")
