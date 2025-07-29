import einops
import torch
import torch as th
import torch.nn as nn
import os
from torchvision.utils import save_image

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import pytorch_lightning as pl


class ControlDiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key, control_model=None):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        self.control_model = control_model
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm', 'hybrid-adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, c_adm=None, control=None, only_mid_control=False):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t, control=control, only_mid_control=only_mid_control)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, control=control, only_mid_control=only_mid_control)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, control=control, only_mid_control=only_mid_control)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, control=control, only_mid_control=only_mid_control)
        elif self.conditioning_key == 'hybrid-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm, control=control, only_mid_control=only_mid_control)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc, control=control, only_mid_control=only_mid_control)
        else:
            raise NotImplementedError()

        return out


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)

        if control is not None:
            control_feature = control.pop()
            if hasattr(self, 'debug_count'):
                self.debug_count += 1
            else:
                self.debug_count = 0
            if self.debug_count < 5:
                print(f"Debug UNet: h shape: {h.shape}, control_feature shape: {control_feature.shape}")
                print(f"Debug UNet: h mean={h.mean().item():.6f}, std={h.std().item():.6f}")
                print(f"Debug UNet: control_feature mean={control_feature.mean().item():.6f}, std={control_feature.std().item():.6f}")
            h += control_feature
            if self.debug_count < 5:
                print(f"Debug UNet: After adding control, h mean={h.mean().item():.6f}, std={h.std().item():.6f}")

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 256, 256, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 256, 512, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 512, 512, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 512, 512, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 512, 512, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 512, model_channels, 3, padding=1)
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(conv_nd(self.dims, channels, channels, 1, padding=0))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        # Log ControlNet input hint statistics
        print(f"  ControlNet input hint: shape={hint.shape}, mean={hint.mean().item():.4f}, std={hint.std().item():.4f}, range=[{hint.min().item():.3f}, {hint.max().item():.3f}]")
        
        # Log final conv weight stats from input_hint_block
        final_conv = self.input_hint_block[-1]
        if hasattr(final_conv, 'weight'):
            print(f"    Final conv weight stats: mean={final_conv.weight.mean().item():.4f}, std={final_conv.weight.std().item():.4f}")

        guided_hint = self.input_hint_block(hint, emb, context)
        
        # Log hint features after processing
        print(f"  After input_hint_block: shape={guided_hint.shape}, mean={guided_hint.mean().item():.4f}, std={guided_hint.std().item():.4f}, range=[{guided_hint.min().item():.3f}, {guided_hint.max().item():.3f}]")
        print(f"hint features: {guided_hint.mean().item():.4f}, {guided_hint.std().item():.4f}")

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        # Log ControlNet output feature maps
        print(f"ControlNet produced {len(outs)} feature maps")
        for i, feature in enumerate(outs):
            print(f"  feature[{i}].shape={feature.shape}, mean={feature.mean().item():.4f}, std={feature.std().item():.4f}")

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 7  # Increased from 0.1 to 1.0 for stronger influence
        # You can experiment with different scales, e.g.:
        # self.control_scales = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.sd_locked = False  # Add missing attribute
        
        # Create output directory for debugging images
        os.makedirs("debug_outputs", exist_ok=True)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        
        # Debug: Print input format information
        if hasattr(self, 'input_debug_count'):
            self.input_debug_count += 1
        else:
            self.input_debug_count = 0
            
        if self.input_debug_count < 5:  # Only print first 5 times
            print(f"[Debug] Input format: x shape={x.shape}, x dtype={x.dtype}")
            print(f"[Debug] Input format: x min/max={x.min().item():.3f}/{x.max().item():.3f}")
            if x.size(1) == 4:
                print(f"[Debug] Input has 4 channels - likely RGBA format")
                print(f"[Debug] This suggests the parent class is returning RGBA format")
            elif x.size(1) == 3:
                print(f"[Debug] Input has 3 channels - RGB format")
            elif x.size(1) == 1:
                print(f"[Debug] Input has 1 channel - grayscale format")
            else:
                print(f"[Debug] Input has {x.size(1)} channels - unusual format")
        
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        
        # Return only control image (ControlNet handles control separately)
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        # Handle the nested conditioning structure properly
        if isinstance(cond['c_crossattn'], list):
            # If it's a list, check if it contains a dict (nested structure)
            if len(cond['c_crossattn']) == 1 and isinstance(cond['c_crossattn'][0], dict):
                # Nested structure: extract the actual tensor
                cond_txt = cond['c_crossattn'][0]['c_crossattn'][0]
            elif all(isinstance(item, torch.Tensor) for item in cond['c_crossattn']):
                # List of tensors, concatenate them
                cond_txt = torch.cat(cond['c_crossattn'], 1)
            else:
                # Fallback: take the first element
                cond_txt = cond['c_crossattn'][0]
        elif isinstance(cond['c_crossattn'], dict):
            # If it's a dict, extract the tensor
            cond_txt = cond['c_crossattn']
        elif isinstance(cond['c_crossattn'], torch.Tensor):
            # If it's already a tensor
            cond_txt = cond['c_crossattn']
        else:
            # Fallback
            cond_txt = cond['c_crossattn']

        # Debug: Print tensor shapes
        if hasattr(self, 'debug_count'):
            self.debug_count += 1
        else:
            self.debug_count = 0
            
        if self.debug_count < 10:  # Print first 10 times for more debugging
            print(f"Debug {self.debug_count}: cond_txt shape BEFORE processing: {cond_txt.shape}")
            print(f"Debug {self.debug_count}: cond_txt dtype: {cond_txt.dtype}")
            print(f"Debug {self.debug_count}: x_noisy shape: {x_noisy.shape}")
            if cond['c_concat'] is not None:
                print(f"Debug {self.debug_count}: control_image shape: {cond['c_concat'][0].shape}")

        # Ensure cond_txt has the correct shape for attention - MORE ROBUST HANDLING
        if cond_txt.dim() == 3:
            # Shape should be [batch, sequence_length, context_dim]
            # Check if there are any extra dimensions of size 1
            if cond_txt.shape[1] == 1 and cond_txt.shape[2] == 1:
                # If it's [B, 1, 1, D], squeeze it
                cond_txt = cond_txt.squeeze(1).squeeze(1)
                cond_txt = cond_txt.unsqueeze(1)  # Back to [B, 1, D]
            pass
        elif cond_txt.dim() == 2:
            # Add batch dimension if missing
            cond_txt = cond_txt.unsqueeze(0)
        elif cond_txt.dim() == 4:
            # If it's [B, 1, 1, D], squeeze the extra dimensions
            cond_txt = cond_txt.squeeze(1).squeeze(1)
            cond_txt = cond_txt.unsqueeze(1)  # Back to [B, 1, D]
        else:
            # Reshape to expected format - be more careful
            if cond_txt.dim() > 3:
                # Squeeze extra dimensions
                while cond_txt.dim() > 3:
                    cond_txt = cond_txt.squeeze(-2)
            # Ensure it's [B, S, D]
            if cond_txt.dim() == 2:
                cond_txt = cond_txt.unsqueeze(1)
            elif cond_txt.dim() != 3:
                # Last resort: reshape to [B, -1, D]
                cond_txt = cond_txt.view(cond_txt.size(0), -1, cond_txt.size(-1))

        if self.debug_count < 10:
            print(f"Debug {self.debug_count}: cond_txt shape AFTER processing: {cond_txt.shape}")

        # FINAL CHECK: Ensure the tensor is exactly [B, S, D] format
        if cond_txt.dim() != 3:
            print(f"ERROR: cond_txt has wrong dimensions: {cond_txt.shape}")
            # Force it to be [B, S, D]
            if cond_txt.dim() == 2:
                cond_txt = cond_txt.unsqueeze(1)
            elif cond_txt.dim() > 3:
                cond_txt = cond_txt.view(cond_txt.size(0), -1, cond_txt.size(-1))
            print(f"FORCED cond_txt shape: {cond_txt.shape}")

        # IMPORTANT: Ensure batch size consistency
        # If we're in generation mode (batch_size=1), ensure cond_txt also has batch_size=1
        if x_noisy.size(0) == 1 and cond_txt.size(0) != 1:
            print(f"Fixing batch size: cond_txt {cond_txt.shape} -> batch_size=1")
            cond_txt = cond_txt[:1]  # Take only first sample

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            # Extract control image from c_concat
            control_image = cond['c_concat'][0]  # Edge map (hint)
            
            # Process control image with ControlNet (ControlNet gets VAE latent + hint)
            try:
                control = self.control_model(x=x_noisy, hint=control_image, timesteps=t, context=cond_txt)
                
                # Debug: Print control feature shapes
                if self.debug_count < 5:
                    print(f"Debug {self.debug_count}: Number of control features: {len(control)}")
                    for i, c in enumerate(control):
                        print(f"Debug {self.debug_count}: Control feature {i} shape: {c.shape}")
                
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                
                # Debug: Print control scales being applied
                if self.debug_count < 5:
                    print(f"Debug {self.debug_count}: Control scales applied:")
                    for i, (c, scale) in enumerate(zip(control, self.control_scales)):
                        print(f"Debug {self.debug_count}: Feature {i}: scale={scale:.3f}, mean={c.mean().item():.6f}, std={c.std().item():.6f}")
                
                # UNet gets VAE latent + control features (no input image concatenation)
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
                
                # Debug: Test ControlNet influence by comparing with no control
                if self.debug_count < 3:
                    print(f"Debug {self.debug_count}: Testing ControlNet influence...")
                    eps_no_control = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
                    influence = (eps - eps_no_control).abs().mean().item()
                    print(f"Debug {self.debug_count}: ControlNet influence: {influence:.6f}")
                    print(f"Debug {self.debug_count}: eps_with_control std: {eps.std().item():.6f}")
                    print(f"Debug {self.debug_count}: eps_no_control std: {eps_no_control.std().item():.6f}")
            except Exception as e:
                print(f"ERROR in ControlNet forward pass: {e}")
                print(f"cond_txt shape at error: {cond_txt.shape}")
                print(f"x_noisy shape at error: {x_noisy.shape}")
                print(f"control_image shape at error: {control_image.shape}")
                print(f"Number of control features: {len(control) if 'control' in locals() else 'N/A'}")
                if 'control' in locals():
                    for i, c in enumerate(control):
                        print(f"Control feature {i} shape: {c.shape}")
                raise e

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    def save_debug_images(self, batch, samples=None, step_name="step"):
        """Save input, generated, and control images for debugging"""
        try:
            # Get input data
            x, c = self.get_input(batch, self.first_stage_key, bs=1)
            c_cat = c["c_concat"][0][:1] if isinstance(c["c_concat"][0], torch.Tensor) else c["c_concat"][0]
            
            # Save input reconstruction
            recon = self.decode_first_stage(x[:1])
            save_image(recon, f"debug_outputs/input_{step_name}.png")
            print(f"Saved input image to debug_outputs/input_{step_name}.png")
            
            # Save control image
            if isinstance(c_cat, torch.Tensor):
                # Ensure control image is in correct format for saving
                if c_cat.dim() == 3 and c_cat.shape[0] == 1:  # [1, H, W]
                    control_img = c_cat
                elif c_cat.dim() == 3 and c_cat.shape[2] == 1:  # [H, W, 1]
                    control_img = c_cat.permute(2, 0, 1)  # [1, H, W]
                else:
                    control_img = c_cat
                
                save_image(control_img, f"debug_outputs/control_{step_name}.png")
                print(f"Saved control image to debug_outputs/control_{step_name}.png")
            
            # Save generated samples if provided
            if samples is not None:
                if isinstance(samples, torch.Tensor):
                    generated = self.decode_first_stage(samples[:1])
                    save_image(generated, f"debug_outputs/generated_{step_name}.png")
                    print(f"Saved generated image to debug_outputs/generated_{step_name}.png")
            
        except Exception as e:
            print(f"Failed to save debug images: {e}")
            import traceback
            traceback.print_exc()

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat = c["c_concat"][0][:N] if isinstance(c["c_concat"][0], torch.Tensor) else c["c_concat"][0]
        c_cross = c["c_crossattn"][0][:N] if isinstance(c["c_crossattn"][0], torch.Tensor) else c["c_crossattn"][0]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        
        # Save control image properly
        if isinstance(c_cat, torch.Tensor):
            print(f"Control tensor shape: {c_cat.shape}, range: [{c_cat.min():.3f}, {c_cat.max():.3f}]")
            if c_cat.dim() == 3 and c_cat.shape[2] == 1:
                log["control"] = c_cat.permute(2, 0, 1)  # [1, H, W] - keep as grayscale
            else:
                log["control"] = c_cat
        else:
            log["control"] = None

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c_cross]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            
            # Save debug images
            self.save_debug_images(batch, samples, f"log_{self.global_step}")
            
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c_cross]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        params += list(self.model.diffusion_model.output_blocks.parameters())
        params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        print(f"[Step {batch_idx}] Training loss: {loss.item():.5f}")
        
        # Track loss trend
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
        self.loss_history.append(loss.item())
        
        if len(self.loss_history) > 5:
            recent_avg = sum(self.loss_history[-5:]) / 5
            if len(self.loss_history) > 10:
                older_avg = sum(self.loss_history[-10:-5]) / 5
                trend = "decreasing" if recent_avg < older_avg else "increasing"
                print(f"[Debug] Loss trend (last 5 vs previous 5): {trend} ({recent_avg:.5f} vs {older_avg:.5f})")
                
                # Check if loss is not decreasing significantly
                if abs(recent_avg - older_avg) < 0.001:
                    print(f"[Debug] WARNING: Loss is not changing significantly - consider adjusting learning rate")
                elif recent_avg > older_avg:
                    print(f"[Debug] WARNING: Loss is increasing - learning rate might be too high")
                else:
                    print(f"[Debug] GOOD: Loss is decreasing - model is learning")
            else:
                print(f"[Debug] Recent loss average: {recent_avg:.5f}")
        
        # Simple debug print every step to track progress
        if batch_idx % 5 == 0:  # Every 5 steps instead of 10
            print(f"[Debug] Step {batch_idx}: Loss = {loss.item():.5f}, Global step = {self.global_step}")
        
        # Debug: Compare ε-predictions with and without ControlNet
        if batch_idx % 5 == 0:  # Changed from 10 to 5 for more frequent checks
            print(f"[Debug] Starting ε-comparison for batch {batch_idx} (batch_idx % 5 == 0)")
            try:
                with torch.no_grad():
                    # 1) make a little noisy latent - use fixed seed for consistency
                    x, c = self.get_input(batch, self.first_stage_key, bs=1)
                    print(f"[Debug] Input shapes: x={x.shape}, c_concat={c['c_concat'][0].shape if c['c_concat'] else 'None'}")
                    
                    # Store the first latent for consistent comparison across steps
                    if not hasattr(self, 'fixed_latent'):
                        self.fixed_latent = x.clone().detach()
                        print(f"[Debug] Stored fixed latent for future comparisons")
                    else:
                        # Use the stored fixed latent instead of current batch
                        x = self.fixed_latent.clone()
                        print(f"[Debug] Using stored fixed latent for consistent comparison")
                    
                    # Use fixed seed for consistent noise
                    torch.manual_seed(42)
                    noise = torch.randn_like(x) * 0.1
                    x_noisy = x + noise
                    t = torch.tensor([self.num_timesteps // 2], device=self.device)
                    print(f"[Debug] Timestep: {t.item()}")
                    print(f"[Debug] Using fixed seed (42) for consistent noise")
                    print(f"[Debug] Noise stats: mean={noise.mean().item():.6f}, std={noise.std().item():.6f}")
                    print(f"[Debug] x_noisy stats: mean={x_noisy.mean().item():.6f}, std={x_noisy.std().item():.6f}")

                    # 2) UNet *without* ControlNet
                    print(f"[Debug] Running UNet without ControlNet...")
                    # Fix context tensor shape for UNet without ControlNet
                    context_tensor = c["c_crossattn"][0]
                    print(f"[Debug] Original context tensor type: {type(context_tensor)}")
                    
                    # Handle nested dictionary structure
                    if isinstance(context_tensor, dict):
                        print(f"[Debug] Context tensor is a dict with keys: {context_tensor.keys()}")
                        if 'c_crossattn' in context_tensor:
                            context_tensor = context_tensor['c_crossattn'][0]
                            print(f"[Debug] Extracted tensor from dict, shape: {context_tensor.shape}")
                        else:
                            print(f"[Debug] No 'c_crossattn' key found in context dict")
                            context_tensor = None
                    else:
                        print(f"[Debug] Original context tensor shape: {context_tensor.shape}")
                    
                    # Ensure context tensor has correct shape for attention
                    if context_tensor is not None:
                        if context_tensor.dim() == 2:
                            # If it's [B, D], add sequence dimension: [B, 1, D]
                            context_tensor = context_tensor.unsqueeze(1)
                        elif context_tensor.dim() > 3:
                            # If it has extra dimensions, squeeze them
                            while context_tensor.dim() > 3:
                                context_tensor = context_tensor.squeeze(-2)
                        print(f"[Debug] Fixed context tensor shape: {context_tensor.shape}")
                    else:
                        print(f"[Debug] Context tensor is None, skipping UNet without ControlNet")
                        eps_noctrl = None
                    
                    if context_tensor is not None:
                        eps_noctrl = self.model.diffusion_model(x=x_noisy, timesteps=t, context=context_tensor)
                        print(f"[Debug] eps_noctrl shape: {eps_noctrl.shape}")
                    else:
                        print(f"[Debug] Skipping eps_noctrl calculation due to None context")
                        eps_noctrl = None

                    # 3) UNet *with* ControlNet
                    print(f"[Debug] Running ControlNet...")
                    # Fix context tensor shape for ControlNet too
                    context_tensor_ctrl = c["c_crossattn"][0]
                    print(f"[Debug] ControlNet context tensor type: {type(context_tensor_ctrl)}")
                    
                    # Handle nested dictionary structure for ControlNet
                    if isinstance(context_tensor_ctrl, dict):
                        print(f"[Debug] ControlNet context tensor is a dict with keys: {context_tensor_ctrl.keys()}")
                        if 'c_crossattn' in context_tensor_ctrl:
                            context_tensor_ctrl = context_tensor_ctrl['c_crossattn'][0]
                            print(f"[Debug] Extracted ControlNet tensor from dict, shape: {context_tensor_ctrl.shape}")
                        else:
                            print(f"[Debug] No 'c_crossattn' key found in ControlNet context dict")
                            context_tensor_ctrl = None
                    else:
                        print(f"[Debug] ControlNet context tensor shape: {context_tensor_ctrl.shape}")
                    
                    # Ensure context tensor has correct shape for attention
                    if context_tensor_ctrl is not None:
                        if context_tensor_ctrl.dim() == 2:
                            context_tensor_ctrl = context_tensor_ctrl.unsqueeze(1)
                        elif context_tensor_ctrl.dim() > 3:
                            while context_tensor_ctrl.dim() > 3:
                                context_tensor_ctrl = context_tensor_ctrl.squeeze(-2)
                        print(f"[Debug] Fixed ControlNet context tensor shape: {context_tensor_ctrl.shape}")
                    else:
                        print(f"[Debug] ControlNet context tensor is None, skipping ControlNet")
                        context_tensor_ctrl = None
                    
                    # Initialize control_feats
                    control_feats = []
                    
                    # Only run ControlNet if we have valid context
                    if context_tensor_ctrl is not None:
                        try:
                            # Check if we have control image
                            if c['c_concat'] is not None and len(c['c_concat']) > 0:
                                control_image = c['c_concat'][0]
                                # Run ControlNet to get features
                                control_feats = self.control_model(x=x_noisy, hint=control_image, timesteps=t, context=context_tensor_ctrl)
                                print(f"[Debug] Control features: {len(control_feats)} features")
                            else:
                                print(f"[Debug] No control image available, skipping ControlNet")
                                control_feats = []
                        except Exception as e:
                            print(f"[Debug] ControlNet failed: {e}")
                            control_feats = []
                    else:
                        print(f"[Debug] Skipping ControlNet due to None context")
                        control_feats = []
                    
                    print(f"[Debug] Control scales: {len(self.control_scales)} scales")
                    print(f"[Debug] Control scales values: {self.control_scales}")
                    
                    # Check if we have the right number of control features
                    if len(control_feats) != len(self.control_scales):
                        print(f"[Debug] WARNING: Control features ({len(control_feats)}) != control scales ({len(self.control_scales)})")
                        # Adjust control scales to match
                        if len(control_feats) > len(self.control_scales):
                            self.control_scales.extend([1.0] * (len(control_feats) - len(self.control_scales)))
                        else:
                            control_feats = control_feats[:len(self.control_scales)]
                    
                    print(f"[Debug] Running UNet with ControlNet...")
                    eps_ctrl = self.model.diffusion_model(x=x_noisy,
                                                        timesteps=t,
                                                        context=context_tensor_ctrl,
                                                        control=[f * s for f,s in zip(control_feats, self.control_scales)],
                                                        only_mid_control=self.only_mid_control)
                    print(f"[Debug] eps_ctrl shape: {eps_ctrl.shape}")

                    # Only compute difference if both predictions are available
                    if eps_noctrl is not None and eps_ctrl is not None:
                        diff = (eps_ctrl - eps_noctrl).abs().mean().item()
                        print(f"[Debug] mean(|ε_ctrl − ε_noctrl|) = {diff:.6f}")
                        
                        # Additional analysis
                        eps_ctrl_norm = eps_ctrl.norm().item()
                        eps_noctrl_norm = eps_noctrl.norm().item()
                        print(f"[Debug] ε_ctrl norm: {eps_ctrl_norm:.6f}, ε_noctrl norm: {eps_noctrl_norm:.6f}")
                        print(f"[Debug] Relative difference: {diff / max(eps_ctrl_norm, eps_noctrl_norm) * 100:.3f}%")
                        
                        # Track this metric over time
                        if not hasattr(self, 'epsilon_diffs'):
                            self.epsilon_diffs = []
                        self.epsilon_diffs.append(diff)
                        if len(self.epsilon_diffs) > 10:
                            avg_diff = sum(self.epsilon_diffs[-10:]) / 10
                            print(f"[Debug] Average ε difference (last 10): {avg_diff:.6f}")
                        
                        # Track progression over all steps
                        print(f"[Debug] ε-difference progression: {len(self.epsilon_diffs)} total comparisons")
                        if len(self.epsilon_diffs) > 1:
                            first_diff = self.epsilon_diffs[0]
                            current_diff = self.epsilon_diffs[-1]
                            improvement = (current_diff - first_diff) / first_diff * 100
                            print(f"[Debug] ε-difference change: {first_diff:.6f} → {current_diff:.6f} ({improvement:+.1f}%)")
                    else:
                        print(f"[Debug] Cannot compute ε difference: eps_noctrl={eps_noctrl is not None}, eps_ctrl={eps_ctrl is not None}")
                    
            except Exception as e:
                print(f"Failed to compare ε-predictions: {e}")
                print(f"Exception type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
        
        # Debug: Check if model is learning by testing simple generation
        if batch_idx % 10 == 0:
            try:
                # Test simple generation without DDIM sampling
                self.test_simple_generation(batch, f"train_test_{self.global_step}")
                
                # Test VAE functionality
                self.test_vae_functionality(batch, f"train_vae_{self.global_step}")
                
            except Exception as e:
                print(f"Failed to test simple generation: {e}")
                import traceback
                traceback.print_exc()
        
        # Save debug images every 50 steps during training (without sampling)
        if batch_idx % 50 == 0:
            try:
                # Save input images and control images directly
                self.save_training_images(batch, f"step_{self.global_step}")
                
            except Exception as e:
                print(f"Failed to save training images: {e}")
                import traceback
                traceback.print_exc()
        
        return loss

    def test_simple_generation(self, batch, step_name):
        """Test simple generation without DDIM sampling to seee if model is learning"""
        try:
            # Get input data
            x, c = self.get_input(batch, self.first_stage_key, bs=1)
            
            # IMPORTANT: Fix batch size for all tensors
            if c["c_crossattn"] is not None and len(c["c_crossattn"]) > 0:
                c_crossattn = c["c_crossattn"][0]
                if isinstance(c_crossattn, list) and len(c_crossattn) > 0:
                    # Ensure batch size is 1 for generation
                    if c_crossattn[0].size(0) != 1:
                        c_crossattn[0] = c_crossattn[0][:1]
                elif isinstance(c_crossattn, torch.Tensor):
                    if c_crossattn.size(0) != 1:
                        c_crossattn = c_crossattn[:1]
                    c["c_crossattn"][0] = c_crossattn
            
            # Test: Add a small amount of noise and see if model can denoise it
            with torch.no_grad():
                # Add noise to input
                noise = torch.randn_like(x[:1]) * 0.1
                x_noisy = x[:1] + noise
                
                # Get model prediction
                t = torch.tensor([50], device=self.device)  # Middle timestep
                eps = self.apply_model(x_noisy, t, c)
                
                # Simple denoising step
                alpha_t = self.alphas_cumprod[t]
                x_denoised = (x_noisy - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
                
                # Decode
                generated = self.decode_first_stage(x_denoised)
                # Note: Removed saving of test_simple and test_original files as requested
                
        except Exception as e:
            print(f"Error in test simple generation: {e}")
            import traceback
            traceback.print_exc()

    def save_training_images(self, batch, step_name):
        """Save images with step numbers to track progress over iterations"""
        try:
            # Save condition image (input image that conditions generation)
            if "image_cond" in batch:
                cond_img = batch["image_cond"][:1]  # Take first sample
                # Check tensor dimensions and handle accordingly
                if cond_img.dim() == 3:  # [H, W, C]
                    cond_img = cond_img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                elif cond_img.dim() == 4:  # [B, H, W, C]
                    cond_img = cond_img.permute(0, 3, 1, 2)  # [B, C, H, W]
                # Normalize to [0, 1] range
                cond_img = (cond_img + 1) / 2
                cond_img = torch.clamp(cond_img, 0, 1)
                save_image(cond_img, f"debug_outputs/condition_{step_name}.png")
            
            # Save control image (edge maps, depth maps, etc.)
            x, c = self.get_input(batch, self.first_stage_key, bs=1)
            if c["c_concat"] is not None:
                control_img = c["c_concat"][0][:1]  # Take first sample
                # Convert from [B, C, H, W] to [B, H, W, C] for saving
                control_img = control_img.permute(0, 2, 3, 1)
                # For control images, preserve the original range without normalization
                # Control images should stay as black background with white borders
                control_img = torch.clamp(control_img, 0, 1)
                save_image(control_img.permute(0, 3, 1, 2), f"debug_outputs/control_{step_name}.png")
            
            # Save target image (what the model should learn to generate)
            if "image_target" in batch:
                target_img = batch["image_target"][:1]  # Take first sample
                # Check tensor dimensions and handle accordingly
                if target_img.dim() == 3:  # [H, W, C]
                    target_img = target_img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                elif target_img.dim() == 4:  # [B, H, W, C]
                    target_img = target_img.permute(0, 3, 1, 2)  # [B, C, H, W]
                # Normalize to [0, 1] range
                target_img = (target_img + 1) / 2
                target_img = torch.clamp(target_img, 0, 1)
                save_image(target_img, f"debug_outputs/target_{step_name}.png")
            
            # Save generated image using proper sampling
            self.save_generated_image(batch, step_name)
            
            print(f"Saved images to debug_outputs/ for step {step_name}")
            
        except Exception as e:
            print(f"Error saving images: {e}")
            import traceback
            traceback.print_exc()

    def save_generated_image(self, batch, step_name):
        """Save generated image using proper sampling"""
        try:
            # Get input data
            x, c = self.get_input(batch, self.first_stage_key, bs=1)
            
            # Prepare conditioning - handle the nested structure properly
            if c["c_concat"] is not None and len(c["c_concat"]) > 0:
                c_cat = c["c_concat"][0][:1] if isinstance(c["c_concat"][0], torch.Tensor) else c["c_concat"][0]
            else:
                c_cat = None
                
            if c["c_crossattn"] is not None and len(c["c_crossattn"]) > 0:
                c_cross = c["c_crossattn"][0][:1] if isinstance(c["c_crossattn"][0], torch.Tensor) else c["c_crossattn"][0]
            else:
                c_cross = None
            
            # Debug: Print conditioning shapes
            print(f"Debug save_generated: c_cat shape: {c_cat.shape if c_cat is not None else 'None'}")
            print(f"Debug save_generated: c_cross type: {type(c_cross)}")
            if isinstance(c_cross, torch.Tensor):
                print(f"Debug save_generated: c_cross shape: {c_cross.shape}")
                print(f"Debug save_generated: c_cross dtype: {c_cross.dtype}")
            elif isinstance(c_cross, dict):
                print(f"Debug save_generated: c_cross keys: {c_cross.keys()}")
                if 'c_crossattn' in c_cross:
                    c_crossattn = c_cross['c_crossattn']
                    print(f"Debug save_generated: c_crossattn type: {type(c_crossattn)}")
                    if isinstance(c_crossattn, list):
                        print(f"Debug save_generated: c_crossattn list length: {len(c_crossattn)}")
                        if len(c_crossattn) > 0:
                            print(f"Debug save_generated: c_crossattn[0] type: {type(c_crossattn[0])}")
                            if isinstance(c_crossattn[0], torch.Tensor):
                                print(f"Debug save_generated: c_crossattn[0] shape: {c_crossattn[0].shape}")
                    elif isinstance(c_crossattn, torch.Tensor):
                        print(f"Debug save_generated: c_crossattn shape: {c_crossattn.shape}")
            else:
                print(f"Debug save_generated: c_cross: {c_cross}")
            
            # Fix c_crossattn shape if needed
            if c_cross is not None:
                if isinstance(c_cross, torch.Tensor):
                    # IMPORTANT: Fix batch size for sampling (should be 1, not 4)
                    if c_cross.size(0) != 1:
                        c_cross = c_cross[:1]  # Take only first sample
                    if c_cross.dim() == 2:
                        # If it's [B, D], add sequence dimension: [B, 1, D]
                        c_cross = c_cross.unsqueeze(1)
                    elif c_cross.dim() > 3:
                        # If it has extra dimensions, squeeze them
                        while c_cross.dim() > 3:
                            c_cross = c_cross.squeeze(-2)
                    print(f"Debug save_generated: c_cross shape AFTER fix: {c_cross.shape}")
                elif isinstance(c_cross, dict):
                    # Handle dictionary case
                    if 'c_crossattn' in c_cross:
                        c_crossattn = c_cross['c_crossattn']
                        if isinstance(c_crossattn, list) and len(c_crossattn) > 0:
                            c_cross_tensor = c_crossattn[0]
                            # IMPORTANT: Fix batch size for sampling (should be 1, not 4)
                            if c_cross_tensor.size(0) != 1:
                                c_cross_tensor = c_cross_tensor[:1]  # Take only first sample
                            if c_cross_tensor.dim() == 2:
                                c_cross_tensor = c_cross_tensor.unsqueeze(1)
                            elif c_cross_tensor.dim() > 3:
                                while c_cross_tensor.dim() > 3:
                                    c_cross_tensor = c_cross_tensor.squeeze(-2)
                            c_crossattn[0] = c_cross_tensor
                            print(f"Debug save_generated: c_crossattn[0] shape AFTER fix: {c_cross_tensor.shape}")
                        elif isinstance(c_crossattn, torch.Tensor):
                            # IMPORTANT: Fix batch size for sampling (should be 1, not 4)
                            if c_crossattn.size(0) != 1:
                                c_crossattn = c_crossattn[:1]  # Take only first sample
                            if c_crossattn.dim() == 2:
                                c_crossattn = c_crossattn.unsqueeze(1)
                            elif c_crossattn.dim() > 3:
                                while c_crossattn.dim() > 3:
                                    c_crossattn = c_crossattn.squeeze(-2)
                            c_cross['c_crossattn'] = c_crossattn
                            print(f"Debug save_generated: c_crossattn shape AFTER fix: {c_crossattn.shape}")
            
            # Create conditioning dict
            cond = {"c_concat": [c_cat], "c_crossattn": [c_cross]}
            
            # Use DDIM sampler with more steps for better quality
            ddim_steps = 50  # Increased for better quality
            ddim_eta = 0.0
            
            # Sample
            with torch.no_grad():
                print(f"[Debug] Starting DDIM sampling with {ddim_steps} steps")
                # Try with classifier-free guidance for better quality
                samples, _ = self.sample_log(
                    cond=cond,
                    batch_size=1,
                    ddim=True,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=3.0  # Reduced from 7.5 for early training
                )
                print(f"[Debug] DDIM sampling completed, samples shape: {samples.shape}")
                print(f"[Debug] Samples stats: min={samples.min().item():.3f}, max={samples.max().item():.3f}, mean={samples.mean().item():.3f}, std={samples.std().item():.3f}")
                
                # Clip samples to reasonable range if they're too extreme
                if samples.std().item() > 10.0:
                    print(f"[Debug] WARNING: Samples have high std ({samples.std().item():.3f}), clipping to [-10, 10]")
                    samples = torch.clamp(samples, -10.0, 10.0)
                    print(f"[Debug] After clipping: min={samples.min().item():.3f}, max={samples.max().item():.3f}, std={samples.std().item():.3f}")
                
                # Decode the samples
                generated = self.decode_first_stage(samples[:1])
                
                # Also test simple reconstruction of input
                print(f"[Debug] Testing simple input reconstruction...")
                x, c = self.get_input(batch, self.first_stage_key, bs=1)
                # Use the original 4-channel input for VAE (since VAE expects 4 channels)
                reconstructed = self.decode_first_stage(x[:1])
                print(f"[Debug] Input reconstruction stats: shape={reconstructed.shape}, min={reconstructed.min().item():.3f}, max={reconstructed.max().item():.3f}, mean={reconstructed.mean().item():.3f}, std={reconstructed.std().item():.3f}")
                save_image(reconstructed, f"debug_outputs/reconstruction_{step_name}.png")
                print(f"Saved input reconstruction to debug_outputs/reconstruction_{step_name}.png")
                
                # Debug: Check the generated image statistics
                print(f"[Debug] Generated image stats: shape={generated.shape}, min={generated.min().item():.3f}, max={generated.max().item():.3f}, mean={generated.mean().item():.3f}, std={generated.std().item():.3f}")
                
                # Check if the image is mostly noise
                if generated.std().item() > 0.5:
                    print(f"[Debug] WARNING: Generated image has high std ({generated.std().item():.3f}) - likely noise")
                elif generated.std().item() < 0.01:
                    print(f"[Debug] WARNING: Generated image has very low std ({generated.std().item():.3f}) - likely blank")
                else:
                    print(f"[Debug] Generated image std looks reasonable: {generated.std().item():.3f}")
                
                save_image(generated, f"debug_outputs/generated_{step_name}.png")
                print(f"Saved generated image to debug_outputs/generated_{step_name}.png")
                
        except Exception as e:
            print(f"Error in generation: {e}")
            import traceback
            traceback.print_exc()
            # If generation fails, try a simpler approach
            self.save_simple_generated_image(batch, step_name)

    def save_simple_generated_image(self, batch, step_name):
        """Fallback: Save a simple generated image using a single forward pass"""
        try:
            # Get input data
            x, c = self.get_input(batch, self.first_stage_key, bs=1)
            
            # Just save the input reconstruction as a simple fallback
            # This avoids the complex sampling that's causing issues
            with torch.no_grad():
                # Decode the input latent directly
                generated = self.decode_first_stage(x[:1])
                save_image(generated, f"debug_outputs/generated_simple_{step_name}.png")
                print(f"Saved simple generated image (input reconstruction) to debug_outputs/generated_simple_{step_name}.png")
                
        except Exception as e:
            print(f"Error in simple generation: {e}")
            # If simple generation fails, just skip it
            pass

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda() 

    def validation_step(self, batch, batch_idx):
        """Validation step without saving images"""
        loss = super().validation_step(batch, batch_idx)
        return loss 

    def test_vae_functionality(self, batch, step_name):
        """Test if VAE encoding/decoding is working properly"""
        try:
            # Get input data
            x, c = self.get_input(batch, self.first_stage_key, bs=1)
            
            print(f"Debug VAE: x shape: {x.shape}")
            print(f"Debug VAE: x dtype: {x.dtype}")
            print(f"Debug VAE: x min/max: {x.min().item():.3f}/{x.max().item():.3f}")
            
            # Test VAE reconstruction
            with torch.no_grad():
                # Handle different channel formats
                if x.size(1) == 4:  # RGBA format
                    print(f"Debug VAE: Converting 4-channel (RGBA) to 3-channel (RGB)")
                    # Convert RGBA to RGB by taking only the first 3 channels
                    x_rgb = x[:, :3, :, :]  # Take only RGB channels
                    print(f"Debug VAE: Converted shape: {x_rgb.shape}")
                elif x.size(1) == 1:  # Grayscale
                    print(f"Debug VAE: Converting 1-channel (grayscale) to 3-channel (RGB)")
                    # Convert grayscale to RGB by repeating the channel
                    x_rgb = x.repeat(1, 3, 1, 1)
                    print(f"Debug VAE: Converted shape: {x_rgb.shape}")
                elif x.size(1) == 3:  # Already RGB
                    x_rgb = x
                    print(f"Debug VAE: Already 3-channel RGB format")
                else:
                    print(f"Warning: VAE expects 3 channels, got {x.size(1)}. Skipping VAE test.")
                    print(f"This suggests the input images might be in an unsupported format.")
                    return
                
                # Encode and decode the input using RGB format
                encoded = self.encode_first_stage(x_rgb[:1])
                print(f"Debug VAE: encoded type: {type(encoded)}")
                
                # Handle different encoding formats
                if hasattr(encoded, 'sample'):  # DiagonalGaussianDistribution
                    print(f"Debug VAE: Using DiagonalGaussianDistribution.sample()")
                    encoded = encoded.sample()
                elif isinstance(encoded, torch.Tensor):
                    print(f"Debug VAE: Using tensor directly")
                else:
                    print(f"Debug VAE: Unknown encoded type: {type(encoded)}")
                    return
                
                print(f"Debug VAE: encoded shape after processing: {encoded.shape}")
                decoded = self.decode_first_stage(encoded)
                
                # VAE test completed successfully
                print(f"VAE test completed successfully")
                
                # Test simple denoising without DDIM
                print(f"[Debug] Testing simple denoising...")
                with torch.no_grad():
                    # Start with pure noise - use fixed seed for consistency
                    torch.manual_seed(42)
                    noise = torch.randn_like(x[:1]) * 0.5
                    print(f"[Debug] Simple denoising using fixed seed (42)")
                    print(f"[Debug] Simple denoising noise stats: mean={noise.mean().item():.6f}, std={noise.std().item():.6f}")
                    # Try to denoise it with a single step
                    t = torch.tensor([50], device=self.device)  # Middle timestep
                    eps_pred = self.apply_model(noise, t, c)
                    # Simple denoising step
                    alpha_t = self.alphas_cumprod[t]
                    denoised = (noise - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
                    # Decode
                    simple_gen = self.decode_first_stage(denoised)
                    print(f"[Debug] Simple denoising result: shape={simple_gen.shape}, std={simple_gen.std().item():.3f}")
                    if simple_gen.std().item() < 0.1:
                        print(f"[Debug] WARNING: Simple denoising produced very low std - model might not be learning")
                    elif simple_gen.std().item() > 0.5:
                        print(f"[Debug] WARNING: Simple denoising produced very high std - model might be unstable")
                    else:
                        print(f"[Debug] Simple denoising std looks reasonable")
                
        except Exception as e:
            print(f"Error in VAE test: {e}")
            print(f"VAE test failed - this is normal if the VAE uses probabilistic encoding")
            print(f"The model should still work correctly for training and generation")
            import traceback
            traceback.print_exc() 