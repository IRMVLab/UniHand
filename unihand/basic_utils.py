"""
Repository: https://github.com/IRMVLab/UniHand
Paper: Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
Authors: Ma et.al.

This file contains how to build diffusion models, encoders, and decoders for Uni-Hand.
"""

from denoising_diffusion import gaussian_diffusion as gd
from denoising_diffusion.gaussian_diffusion import SpacedDiffusion, space_timesteps
from denoising_diffusion.denoising_models import HOIMamba_homo, HOIMambaTransformer
from denoising_diffusion.networks import PreEncoder, MotionEncoder, LocEncoder, GLIPEncoder, VoxelEncoder, OccFeatEncoder, PostEncoder, ContactEncoder

def create_model_and_diffusion(
    model_cfg,
    diffusion_steps,
    noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    **kwargs,
):

    pre_enc_cfg = model_cfg.get('pre_encoder', {})
    post_enc_cfg = model_cfg.get('post_encoder', {})
    motion_enc_cfg = model_cfg.get('motion_encoder', {})
    loc_enc_cfg = model_cfg.get('loc_encoder', {})
    glip_enc_cfg = model_cfg.get('glip_encoder', {})
    voxel_enc_cfg = model_cfg.get('voxel_encoder', {})
    occ_feat_enc_cfg = model_cfg.get('occ_feat_encoder', {})
    contact_enc_cfg = model_cfg.get('contact_encoder', {})
    denoise_cfg = model_cfg.get('denoised_model', {})

    pre_encoder = PreEncoder(
        input_dims=pre_enc_cfg.get("input_dims", 2049),
        output_dims=pre_enc_cfg.get("output_dims", 1024),
        encoder_hidden_dims=pre_enc_cfg.get("encoder_hidden_dims", 64)
    )
    
    post_encoder = PostEncoder(
        input_dims=post_enc_cfg.get("input_dims", 1024),
        output_dims=post_enc_cfg.get("output_dims", 3),
        encoder_hidden_dims1=post_enc_cfg.get("encoder_hidden_dims1", 256),
        encoder_hidden_dims2=post_enc_cfg.get("encoder_hidden_dims2", 64)
    )
    
    motion_encoder = MotionEncoder(
        input_dims=motion_enc_cfg.get("input_dims", 9),
        output_dims=motion_enc_cfg.get("output_dims", 1024),
        encoder_hidden_dims=motion_enc_cfg.get("encoder_hidden_dims", 64)
    )
    
    loc_encoder = LocEncoder(
        loc_enc_cfg.get("input_dims", 3),
        hidden_features=loc_enc_cfg.get("hidden_features", 256),
        out_features=loc_enc_cfg.get("out_features", 1024)
    )
    
    glip_encoder = GLIPEncoder(
        input_dims=glip_enc_cfg.get("input_dims", 768),
        output_dims=glip_enc_cfg.get("output_dims", 1024)
    )
    
    voxel_encoder = VoxelEncoder(
        input_dims=voxel_enc_cfg.get("input_dims", 1),
        output_dims=voxel_enc_cfg.get("output_dims", 64)
    )
    
    occ_feat_encoder = OccFeatEncoder(
        input_dims=occ_feat_enc_cfg.get("input_dims", 64),
        output_dims=occ_feat_enc_cfg.get("output_dims", 1024),
        encoder_hidden_dims=occ_feat_enc_cfg.get("encoder_hidden_dims", 64)
    )
    
    contact_encoder = ContactEncoder(
        input_dims=contact_enc_cfg.get("input_dims", 1024),
        output_dims=contact_enc_cfg.get("output_dims", 1),
        encoder_hidden_dims1=contact_enc_cfg.get("encoder_hidden_dims1", 256),
        encoder_hidden_dims2=contact_enc_cfg.get("encoder_hidden_dims2", 64)
    )

    denoised_model = HOIMambaTransformer(
        d_model=denoise_cfg.get("d_model", 1024),
        n_layers=denoise_cfg.get("n_layers", 6)
    )


    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas = learn_sigma,
        sigma_small = sigma_small,
        use_kl = use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return pre_encoder, denoised_model, diffusion, post_encoder, motion_encoder, loc_encoder, glip_encoder,voxel_encoder, occ_feat_encoder, contact_encoder


def homo_create_model_and_diffusion(
    model_cfg,
    diffusion_steps,
    noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    **kwargs,
):

    homo_denoise_cfg = model_cfg.get('homo_denoised_model', {})
    denoised_model = HOIMamba_homo(
        d_model=homo_denoise_cfg.get("d_model", 1024),
        n_layers=homo_denoise_cfg.get("n_layers", 6)
    )
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas = learn_sigma,
        sigma_small = sigma_small,
        use_kl = use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return denoised_model, diffusion