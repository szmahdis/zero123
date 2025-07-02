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

def run_example():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = OmegaConf.load("configs/sd-objaverse-finetune-c_concat-256.yaml")
    # Load the main model
    turncam = load_model_from_config(config, "105000.ckpt", device)
    carvekit = create_carvekit_interface()
    nsfw = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to(device)
    clip_fe = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")

    models = {"turncam": turncam, "carvekit": carvekit, "nsfw": nsfw, "clip_fe": clip_fe}

    img = Image.open("apple.png").convert("RGBA")

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
        x=0.0, y=90.0, z=0.0,  # view angles
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
    run_example()
