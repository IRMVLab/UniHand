"""
Repository: https://github.com/IRMVLab/UniHand
Paper: Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
Authors: Ma et.al.

This file contains classes of denoising models for HTP diffusion and Egomotion diffusion.
"""

import torch
from mambapy.mamba import Mamba, MambaConfig, Mamba_homo

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)

from einops import rearrange

def get_pad_mask(seq, pad_idx=0):
    """
    Generate padding mask for sequence
    :param seq: Input sequence tensor
    :param pad_idx: Padding index value
    :return: Boolean mask indicating non-padding positions
    """
    if seq.dim() != 2:
        raise ValueError("<seq> has to be a 2-dimensional tensor!")
    if not isinstance(pad_idx, int):
        raise TypeError("<pad_index> has to be an int!")

    return (seq != pad_idx).unsqueeze(1) # equivalent (seq != pad_idx).unsqueeze(-2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    :param x: Input tensor
    :param drop_prob: Probability of dropping a path
    :param training: Whether the model is in training mode
    :return: Output tensor with dropped paths
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with different dimensional tensors
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() # Binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class Mlp(nn.Module):
    """
    MLP layer with GELU activation
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism
    """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention mechanism
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
           self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
           self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attention = ScaledDotProductAttention(temperature=qk_scale or head_dim ** 0.5)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v, mask=None):
        B, Nq, Nk, Nv, C = q.shape[0], q.shape[1], k.shape[1], v.shape[1], q.shape[2]
        if self.with_qkv:
            q = self.proj_q(q).reshape(B, Nq, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = self.proj_k(k).reshape(B, Nk, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = self.proj_v(v).reshape(B, Nv, self.num_heads, C // self.num_heads).transpose(1, 2)
        else:
            q = q.reshape(B, Nq, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = k.reshape(B, Nk, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = v.reshape(B, Nv, self.num_heads, C // self.num_heads).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)

        x, attn = self.attention(q, k, v, mask=mask)
        x = x.transpose(1, 2).reshape(B, Nq, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    """
    Transformer decoder block with self-attention and cross-attention
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.enc_dec_attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                               qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, tgt, emb_motion, memory_mask=None, trg_mask=None):
        tgt_2 = self.norm1(tgt)
        # Cross-attention with motion features
        tgt = tgt + self.drop_path(self.enc_dec_attn(q=self.norm2(tgt), k=emb_motion, v=emb_motion, mask=None))
        tgt = tgt + self.drop_path(self.mlp(self.norm2(tgt)))
        return tgt

class EncoderBlock(nn.Module):
    """
    Transformer encoder block
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, N, mask=None):
        if mask is not None:
            src_mask = rearrange(mask, 'b n t -> b (n t)', b=B, n=N, t=T)
            src_mask = get_pad_mask(src_mask, 0)
        else:
            src_mask = None
        x2 = self.norm1(x)
        x = x + self.drop_path(self.attn(q=x2, k=x2, v=x2, mask=src_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x




class HOIMamba_homo(nn.Module):
    """
    Homogeneous Mamba model with timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        d_model=1024,
        n_layers=1,
        input_dims=1024,
        output_dims=1024,
        hidden_t_dim=1024,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        config = MambaConfig(d_model=self.d_model, n_layers=self.n_layers)
        self.mamba_encoder = Mamba_homo(config)

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.hidden_size = 1024

        time_embed_dim = hidden_t_dim * 2
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, self.hidden_size),
        )

        self.register_buffer("position_ids", torch.arange(1000).expand((1, -1)))
        self.position_embeddings = nn.Embedding(1000, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
    

        if self.output_dims != self.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                                nn.Tanh(), nn.Linear(self.hidden_size, self.output_dims))


    def forward(self, x_r, timesteps, motion_feat_encoded, valid_mask=None):
        """
        Forward pass of the homogeneous Mamba model
        :param x_r: Input tensor [B, T, D]
        :param timesteps: Timestep information for diffusion
        :param motion_feat_encoded: Encoded motion features
        :param valid_mask: Mask for valid positions
        :return: Transformed output tensor
        """

        valid_mask = None # Disable masking
        if valid_mask is not None:
            valid_mask = get_pad_mask(valid_mask, pad_idx=0)

        # Time embedding
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        emb_xr = x_r

        seq_length = x_r.size(1)
        position_ids = self.position_ids[:, : seq_length]
        # Combine positional, input, and time embeddings
        emb_inputs_r = self.position_embeddings(position_ids) + emb_xr + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs_r = self.LayerNorm(emb_inputs_r)

        # Motion feature encoding
        motion_length = motion_feat_encoded.size(1)
        position_ids_motion = self.position_ids[:, : motion_length]
        emb_motion = self.position_embeddings(position_ids_motion) + motion_feat_encoded
        emb_motion = self.LayerNorm(emb_motion)

        # Mamba processing
        emb_inputs_r = self.mamba_encoder(emb_inputs_r, emb_motion)

        emb_inputs_r = self.LayerNorm(emb_inputs_r)

        # Project back to output dimension
        if self.output_dims != self.hidden_size:
            h_r = self.output_down_proj(emb_inputs_r)
        else:
            h_r = emb_inputs_r
        h_r = h_r.type(x_r.dtype)

        return h_r


class HOIMambaTransformer(nn.Module):
    """
    Hybrid Mamba-Transformer model with timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        d_model=1024,
        n_layers=1,
        input_dims=1024,
        output_dims=1024,
        hidden_t_dim=1024,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        config = MambaConfig(d_model=self.d_model, n_layers=self.n_layers)
        self.mamba_encoder = Mamba(config)
        self.mamba_encoder_add = Mamba(config)

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.hidden_size = 1024

        time_embed_dim = hidden_t_dim * 2
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, self.hidden_size),
        )

        self.register_buffer("position_ids", torch.arange(1000).expand((1, -1)))
        self.position_embeddings = nn.Embedding(1000, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
    
        # Hybrid architecture with 1 transformer layer
        self.dropout_value=0.1
        drop_path_rate = 0.1
        self.depth = 1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]

        self.denoised_transformer = nn.ModuleList([DecoderBlock(dim=self.hidden_size, num_heads=4, mlp_ratio=4, qkv_bias=False, qk_scale=None,
        drop=self.dropout_value, attn_drop=0., drop_path=dpr[i])
        for i in range(self.depth)])

        if self.output_dims != self.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                                nn.Tanh(), nn.Linear(self.hidden_size, self.output_dims))


    def forward(self, x_r, timesteps, motion_feat_encoded_raw, valid_mask=None):
        """
        Forward pass of the hybrid Mamba-Transformer model
        :param x_r: Input tensor [B, T, D]
        :param timesteps: Timestep information for diffusion
        :param motion_feat_encoded_raw: List of encoded features [motion_feat, occ_feat]
        :param valid_mask: Mask for valid positions
        :return: Transformed output tensor
        """

        # Extract motion and occupancy features
        motion_feat_encoded = motion_feat_encoded_raw[0]
        occ_feat_encoded =  motion_feat_encoded_raw[1]

        valid_mask = None # Disable masking
        if valid_mask is not None:
            valid_mask = get_pad_mask(valid_mask, pad_idx=0)

        # Time embedding
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))
        emb_xr = x_r

        seq_length = x_r.size(1)
        position_ids = self.position_ids[:, : seq_length]
        # Combine positional, input, and time embeddings
        emb_inputs_r = self.position_embeddings(position_ids) + emb_xr + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs_r = self.LayerNorm(emb_inputs_r)

        # Occupancy feature encoding
        occ_length = occ_feat_encoded.size(1)
        position_ids_occ = self.position_ids[:, : occ_length]
        emb_occ = self.position_embeddings(position_ids_occ) + occ_feat_encoded
        emb_occ = self.LayerNorm(emb_occ)

        # First Mamba block
        emb_inputs_r = self.mamba_encoder(emb_inputs_r, emb_inputs_r)
        emb_inputs_r = self.LayerNorm(emb_inputs_r)

        # First Transformer block with occupancy attention
        for blk in self.denoised_transformer:
            emb_inputs_r = emb_inputs_r + blk(emb_inputs_r, emb_occ, memory_mask=None, trg_mask=valid_mask)

        # Project back to output dimension
        if self.output_dims != self.hidden_size:
            h_r = self.output_down_proj(emb_inputs_r)
        else:
            h_r = emb_inputs_r
        h_r = h_r.type(x_r.dtype)

        return h_r