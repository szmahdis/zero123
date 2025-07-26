#!/usr/bin/env python3

import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange
import cv2
import random

from ldm.modules.controlnet import ControlLDM
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def seed_everything(seed=None):
    """Set random seed for reproducibility"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path, size=256):
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.LANCZOS)
    image = np.array(image)
    image = image.astype(np.float32) / 127.5 - 1.0
    return image


def load_control_img(path, size=256, channels=3):
    """Load control image/feature map"""
    if channels == 1:
        image = Image.open(path).convert("L")
    else:
        image = Image.open(path).convert("RGB")
    
    image = image.resize((size, size), Image.LANCZOS)
    image = np.array(image)
    image = image.astype(np.float32) / 127.5 - 1.0
    
    if channels == 1:
        image = image[..., None]
    
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a beautiful landscape",
        help="the prompt to render"
    )
    parser.add_argument(
        "--control_path",
        type=str,
        required=True,
        help="path to control image/feature map"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/controlnet"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/controlnet_zero123.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="105000.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--control_channels",
        type=int,
        default=3,
        help="number of channels in control image (1 for grayscale, 3 for RGB, etc.)"
    )
    parser.add_argument(
        "--only_mid_control",
        action='store_true',
        help="only apply control to middle block"
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/zero123_controlnet.ckpt"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampling not implemented for ControlNet")

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WMarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # Load control image
    control_image = load_control_img(opt.control_path, size=opt.H, channels=opt.control_channels)
    control_image = torch.from_numpy(control_image).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)

    print(f"running inference for {len(data)} prompts")
    for n in range(opt.n_iter):
        for prompts in tqdm(data, desc=f"Generating images for iteration {n+1}/{opt.n_iter}"):
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            c = model.get_learned_conditioning(prompts)

            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            samples_ddim, _ = sampler.sample(
                S=opt.ddim_steps,
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=opt.scale,
                unconditional_conditioning=uc,
                eta=opt.ddim_eta,
                x_T=start_code,
                control=control_image,
                only_mid_control=opt.only_mid_control
            )

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

            x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

            if not opt.skip_save:
                for x_sample in x_checked_image_torch:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1

            if not opt.skip_grid:
                # also, sample as before and get unconditioned code
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)

                samples_cfg, _ = sampler.sample_log(
                    S=opt.ddim_steps,
                    conditioning=c,
                    batch_size=batch_size,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uc,
                    eta=opt.ddim_eta,
                    x_T=start_code,
                    control=control_image,
                    only_mid_control=opt.only_mid_control
                )
                x_samples_cfg = model.decode_first_stage(samples_cfg)
                x_samples_cfg = torch.clamp((x_samples_cfg + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_cfg = x_samples_cfg.cpu().permute(0, 2, 3, 1).numpy()

                x_checked_image, has_nsfw_concept = check_safety(x_samples_cfg)
                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                def uncond_conditioning(shape, uc, batch_size):
                    uc_tile = uc.repeat(batch_size, 1, 1)
                    return uc_tile

                def do_plot():
                    images = torch.stack([x_checked_image_torch[i] for i in range(batch_size)])
                    grid = torchvision.utils.make_grid(images, nrow=n_rows)
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    wm = put_watermark(Image.fromarray(grid.astype(np.uint8)), wm_encoder)
                    wm.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                do_plot()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")


if __name__ == "__main__":
    main() 