import torch
import torch.nn as nn

from .utils.nn import SiLU, linear, timestep_embedding


class TrajectoryDiT(nn.Module):
    """
    Trajectory-space DiT inspired by diffusion_policy's transformer diffusion model.

    The diffusion sample is the trajectory itself, while conditioning tokens are built
    from temporal visual / past-trajectory features and 3D occupancy features.
    """

    def __init__(
        self,
        input_dims=3,
        output_dims=3,
        cond_dims=1024,
        occ_cond_dims=1024,
        d_model=512,
        n_layers=6,
        n_heads=8,
        hidden_t_dim=256,
        max_seq_len=40,
        max_occ_tokens=64,
        n_cond_layers=2,
        dropout=0.1,
        attn_dropout=0.1,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.cond_dims = cond_dims
        self.occ_cond_dims = occ_cond_dims
        self.d_model = d_model
        self.hidden_t_dim = hidden_t_dim
        self.max_seq_len = max_seq_len
        self.max_occ_tokens = max_occ_tokens

        self.sample_proj = nn.Linear(input_dims, d_model)
        self.temporal_cond_proj = nn.Linear(cond_dims, d_model)
        self.occ_cond_proj = nn.Linear(occ_cond_dims, d_model)
        self.observed_mask_proj = nn.Linear(1, d_model)

        self.sample_pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.temporal_pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.occ_pos_emb = nn.Parameter(torch.zeros(1, max_occ_tokens, d_model))

        time_embed_dim = hidden_t_dim * 2
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, d_model),
        )

        self.input_drop = nn.Dropout(dropout)
        self.cond_drop = nn.Dropout(dropout)

        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=attn_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.cond_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_cond_layers,
            )
        else:
            self.cond_encoder = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layers,
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, output_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.sample_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.temporal_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.occ_pos_emb, mean=0.0, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name, None)
                if weight is not None:
                    nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name, None)
                if bias is not None:
                    nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _build_memory(
        self,
        timesteps,
        temporal_cond,
        occ_cond,
        observed_mask,
        valid_mask,
    ):
        batch_size = timesteps.shape[0]
        time_token = self.time_embed(
            timestep_embedding(timesteps, self.hidden_t_dim)
        ).unsqueeze(1)

        cond_tokens = [time_token]
        memory_key_padding_mask = [
            torch.zeros(batch_size, 1, device=timesteps.device, dtype=torch.bool)
        ]

        if temporal_cond is not None:
            time_steps = temporal_cond.shape[1]
            temporal_tokens = self.temporal_cond_proj(temporal_cond)
            temporal_tokens = temporal_tokens + self.temporal_pos_emb[:, :time_steps]
            if observed_mask is not None:
                temporal_tokens = temporal_tokens + self.observed_mask_proj(
                    observed_mask[:, :time_steps].unsqueeze(-1).float()
                )
            cond_tokens.append(temporal_tokens)
            if valid_mask is None:
                memory_key_padding_mask.append(
                    torch.zeros(
                        batch_size,
                        time_steps,
                        device=timesteps.device,
                        dtype=torch.bool,
                    )
                )
            else:
                memory_key_padding_mask.append(~valid_mask[:, :time_steps].bool())

        if occ_cond is not None:
            occ_steps = occ_cond.shape[1]
            occ_tokens = self.occ_cond_proj(occ_cond)
            occ_tokens = occ_tokens + self.occ_pos_emb[:, :occ_steps]
            cond_tokens.append(occ_tokens)
            memory_key_padding_mask.append(
                torch.zeros(
                    batch_size,
                    occ_steps,
                    device=timesteps.device,
                    dtype=torch.bool,
                )
            )

        memory = torch.cat(cond_tokens, dim=1)
        memory = self.cond_drop(memory)
        memory_key_padding_mask = torch.cat(memory_key_padding_mask, dim=1)

        if isinstance(self.cond_encoder, nn.TransformerEncoder):
            memory = self.cond_encoder(
                memory,
                src_key_padding_mask=memory_key_padding_mask,
            )
        else:
            memory = self.cond_encoder(memory)

        return memory, memory_key_padding_mask

    def forward(self, x_r, timesteps, condition, valid_mask=None):
        temporal_cond = condition.get("temporal_cond")
        occ_cond = condition.get("occ_cond")
        observed_mask = condition.get("observed_mask")

        batch_size, seq_len = x_r.shape[:2]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}."
            )

        if observed_mask is not None and observed_mask.shape[1] != seq_len:
            raise ValueError("observed_mask length must match the trajectory length.")

        sample_tokens = self.sample_proj(x_r)
        sample_tokens = sample_tokens + self.sample_pos_emb[:, :seq_len]
        if observed_mask is not None:
            sample_tokens = sample_tokens + self.observed_mask_proj(
                observed_mask.unsqueeze(-1).float()
            )
        sample_tokens = self.input_drop(sample_tokens)

        if valid_mask is None:
            tgt_key_padding_mask = None
        else:
            tgt_key_padding_mask = ~valid_mask.bool()

        memory, memory_key_padding_mask = self._build_memory(
            timesteps=timesteps,
            temporal_cond=temporal_cond,
            occ_cond=occ_cond,
            observed_mask=observed_mask,
            valid_mask=valid_mask,
        )

        hidden = self.decoder(
            tgt=sample_tokens,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        hidden = self.out_norm(hidden)
        output = self.out_head(hidden).type(x_r.dtype)

        if valid_mask is not None:
            output = output * valid_mask.unsqueeze(-1).to(output.dtype)

        return output
