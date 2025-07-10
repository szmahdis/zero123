import torch
from PIL import Image
from gradio_new import load_model_from_config, main_run, create_carvekit_interface
from ldm.models.diffusion.ddim import DDIMSampler
from transformers import AutoFeatureExtractor
from gradio_new import StableDiffusionSafetyChecker
from omegaconf import OmegaConf
import torchvision.transforms as transforms
from einops import rearrange
import numpy as np
import argparse

def run_inference_on_one_image(path_to_image, x_angle, y_angle, z_angle):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    config_path = "configs/sd-objaverse-finetune-c_concat-256.yaml"
    print(f"config_path: {config_path}")
    config = OmegaConf.load(config_path)

    # Load the main model
    turncam = load_model_from_config(config, "105000.ckpt", device)
    carvekit = create_carvekit_interface()
    nsfw = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to(device)
    clip_fe = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")

    models = {"turncam": turncam, "carvekit": carvekit, "nsfw": nsfw, "clip_fe": clip_fe}

    print(f"path_to_image: {path_to_image}")
    img = Image.open(path_to_image).convert("RGBA")

    # Run generation
    description, _fig, preproc, output_images = main_run(
        models=models,
        device=device,
        cam_vis=type("CV", (), {"polar_change": lambda *a: None,
                                "azimuth_change": lambda *a: None,
                                "radius_change": lambda *a: None,
                                "encode_image": lambda *a: None,
                                "update_figure": lambda *a: None}),
        return_what="gen",
        x=x_angle, y=y_angle, z=z_angle,
        raw_im=img,
        preprocess=True,
        scale=3.0,
        n_samples=2,
        ddim_steps=50,
        ddim_eta=1.0,
        precision="fp32",
        h=256,
        w=256,
    )

    # Save generated images
    for idx, out in enumerate(output_images):
        out.save(f"gen_output_{idx}.png")
    print("Saved generated images.")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Run Zero123 inference on an input image")
    parser.add_argument(
        "-i", "--image", required=True,
        help="path to the input image",
        type=str
    )
    parser.add_argument(
        "--x", type=float, default=0.0,
        help="camera angle x (default: 0.0)"
    )
    parser.add_argument(
        "--y", type=float, default=90.0,
        help="camera angle y (default: 90.0)"
    )
    parser.add_argument(
        "--z", type=float, default=0.0,
        help="camera angle z (default: 0.0)"
    )
    args = parser.parse_args()

    run_inference_on_one_image(args.image, args.x, args.y, args.z)

