'''
conda activate zero123
cd stable-diffusion
python gradio_new_controlnet.py 0
'''

## ControlNet-based Zero123 with Canny edge detection ##

import diffusers  # 0.12.1
import math
import fire
import gradio as gr
import lovely_numpy
import lovely_tensors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import rich
import sys
import time
import torch
import cv2
import os
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from functools import partial
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from transformers import AutoFeatureExtractor
from torch import autocast
from torchvision import transforms


_SHOW_DESC = True
_SHOW_INTERMEDIATE = False
# _SHOW_INTERMEDIATE = True
_GPU_INDEX = 0
# _GPU_INDEX = 2

# _TITLE = 'Zero-Shot Control of Camera Viewpoints within a Single Image'
_TITLE = 'Zero-1-to-3 ControlNet: Edge Map Controlled Novel View Generation'

# This demo allows you to generate novel viewpoints of an object depicted in an input image using ControlNet with edge maps.
_DESCRIPTION = '''
This demo allows you to control camera rotation and thereby generate novel viewpoints of an object within a single image using ControlNet with edge map conditioning.
It is based on Stable Diffusion with ControlNet integration. Check out our [project webpage](https://zero123.cs.columbia.edu/) and [paper](https://arxiv.org/) if you want to learn more about the method!
Note that this model is not intended for images of humans or faces, and is unlikely to work well for them.
'''

_ARTICLE = 'See uses.md'


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    
    # Debug: Check if ControlNet is properly loaded
    if hasattr(model, 'control_model'):
        print(f"ControlNet loaded successfully: {type(model.control_model)}")
        print(f"ControlNet parameters: {sum(p.numel() for p in model.control_model.parameters())}")
    else:
        print("Warning: No ControlNet found in model!")
    
    return model


def create_edge_control(image_path):
    """Create edge-based control image from input image using Canny edge detection (matching generate_control_images.py)."""
    # Read image
    if isinstance(image_path, str):
        # Load with OpenCV if it's a file path
        image = cv2.imread(image_path)
    else:
        # Convert PIL image to OpenCV format
        if hasattr(image_path, 'convert') and image_path.mode == 'RGBA':
            image_path = image_path.convert('RGB')
        
        # Convert PIL to numpy array and then to BGR for OpenCV
        pil_array = np.array(image_path)
        if len(pil_array.shape) == 3:
            # Convert RGB to BGR
            image = pil_array[:, :, ::-1]  # Reverse channels
        else:
            image = pil_array
    
    if image is None:
        print(f"Failed to load image")
        return None
    
    # Convert to grayscale (same as original script)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise (same as original script)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny (same parameters as original script)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Convert to PIL Image
    edges_pil = Image.fromarray(edges, mode='L')
    
    return edges_pil


@torch.no_grad()
def sample_model_controlnet(input_im, control_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                           ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            # Get CLIP conditioning from input image
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            
            # Add camera parameters
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            
            # Prepare conditioning dictionary for ControlNet
            cond = {}
            cond['c_crossattn'] = [c]
            
            # Process control image (edge map)
            if control_im is not None:
                # Convert control image to tensor and normalize to [-1, 1] range
                control_tensor = transforms.ToTensor()(control_im).unsqueeze(0).to(c.device)
                control_tensor = control_tensor * 2 - 1  # Convert from [0,1] to [-1,1]
                control_tensor = control_tensor.repeat(n_samples, 1, 1, 1)
                # Keep control image at original resolution (256x256) - ControlNet will handle downsampling
                cond['c_concat'] = [control_tensor]
                print(f"Control tensor shape: {control_tensor.shape}, range: [{control_tensor.min():.3f}, {control_tensor.max():.3f}]")
                print(f"Control image mode: {control_im.mode}, size: {control_im.size}")
            else:
                # Fallback to original method if no control image
                cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                    .repeat(n_samples, 1, 1, 1)]
            
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 1, 256, 256).to(c.device)]  # 1 channel for edge maps at original resolution
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            # Use the ControlLDM's sample_log method instead of DDIM sampler directly
            samples_ddim, _ = model.sample_log(cond=cond,
                                             batch_size=n_samples,
                                             ddim=True,
                                             ddim_steps=ddim_steps,
                                             eta=ddim_eta,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc)
            
            print(samples_ddim.shape)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


class CameraVisualizer:
    def __init__(self, gradio_plot):
        self._gradio_plot = gradio_plot
        self._fig = None
        self._polar = 0.0
        self._azimuth = 0.0
        self._radius = 0.0
        self._raw_image = None
        self._8bit_image = None
        self._image_colorscale = None

    def polar_change(self, value):
        self._polar = value

    def azimuth_change(self, value):
        self._azimuth = value

    def radius_change(self, value):
        self._radius = value

    def encode_image(self, raw_image):
        '''
        :param raw_image (H, W, 3) array of uint8 in [0, 255].
        '''
        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype='uint8')).convert('P', palette='WEB')
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

        self._raw_image = raw_image
        self._8bit_image = Image.fromarray(raw_image).convert('P', palette='WEB', dither=None)
        self._image_colorscale = [
            [i / 255.0, 'rgb({}, {}, {})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]

    def update_figure(self):
        fig = go.Figure()

        if self._raw_image is not None:
            (H, W, C) = self._raw_image.shape

            x = np.zeros((H, W))
            (y, z) = np.meshgrid(np.linspace(-1.0, 1.0, W), np.linspace(1.0, -1.0, H) * H / W)
            print('x:', lo(x))
            print('y:', lo(y))
            print('z:', lo(z))

            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                surfacecolor=self._8bit_image,
                cmin=0,
                cmax=255,
                colorscale=self._image_colorscale,
                showscale=False,
                lighting_diffuse=1.0,
                lighting_ambient=1.0,
                lighting_fresnel=1.0,
                lighting_roughness=1.0,
                lighting_specular=0.3))

            scene_bounds = 3.5
            base_radius = 2.5
            zoom_scale = 1.5
            fov_deg = 50.0
            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

            input_cone = calc_cam_cone_pts_3d(
                0.0, 0.0, base_radius, fov_deg)
            output_cone = calc_cam_cone_pts_3d(
                self._polar, self._azimuth, base_radius + self._radius * zoom_scale, fov_deg)

            for (cone, clr, legend) in [(input_cone, 'green', 'Input view'),
                                        (output_cone, 'blue', 'Target view')]:

                for (i, edge) in enumerate(edges):
                    (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                    (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                    (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                    fig.add_trace(go.Scatter3d(
                        x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                        line=dict(color=clr, width=3),
                        name=legend, showlegend=(i == 0)))

                if cone[0, 2] <= base_radius / 2.0:
                    fig.add_trace(go.Scatter3d(
                        x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] - 0.05], showlegend=False,
                        mode='text', text=legend, textposition='bottom center'))
                else:
                    fig.add_trace(go.Scatter3d(
                        x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] + 0.05], showlegend=False,
                        mode='text', text=legend, textposition='top center'))

            fig.update_layout(
                height=360,
                autosize=True,
                hovermode=False,
                margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                showlegend=True,
                legend=dict(
                    yanchor='bottom',
                    y=0.01,
                    xanchor='right',
                    x=0.99,
                ),
                scene=dict(
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=1.0),
                    camera=dict(
                        eye=dict(x=base_radius - 1.6, y=0.0, z=0.6),
                        center=dict(x=0.0, y=0.0, z=0.0),
                        up=dict(x=0.0, y=0.0, z=1.0)),
                    xaxis_title='',
                    yaxis_title='',
                    zaxis_title='',
                    xaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    yaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    zaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks='')))

        self._fig = fig
        return fig


def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0

        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    print('new input_im:', lo(input_im))

    return input_im


def main_run_controlnet(models, device, cam_vis, return_what,
                        x=0.0, y=0.0, z=0.0,
                        raw_im=None, control_im=None, preprocess=True,
                        scale=3.0, n_samples=4, ddim_steps=50, ddim_eta=1.0,
                        precision='fp32', h=256, w=256):
    '''
    :param raw_im (PIL Image).
    :param control_im (PIL Image) - edge map control image.
    '''
    
    raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
    safety_checker_input = models['clip_fe'](raw_im, return_tensors='pt').to(device)
    (image, has_nsfw_concept) = models['nsfw'](
        images=np.ones((1, 3)), clip_input=safety_checker_input.pixel_values)
    print('has_nsfw_concept:', has_nsfw_concept)
    if np.any(has_nsfw_concept):
        print('NSFW content detected.')
        to_return = [None] * 10
        description = ('###  <span style="color:red"> Unfortunately, '
                       'potential NSFW content was detected, '
                       'which is not supported by our model. '
                       'Please try again with a different image. </span>')
        if 'angles' in return_what:
            to_return[0] = 0.0
            to_return[1] = 0.0
            to_return[2] = 0.0
            to_return[3] = description
        else:
            to_return[0] = description
        return to_return

    else:
        print('Safety check passed.')

    input_im = preprocess_image(models, raw_im, preprocess)

    # Generate control image if not provided
    if control_im is None:
        control_im = create_edge_control(raw_im)
        if control_im is None:
            print("Failed to create control image")
            return None

    show_in_im1 = (input_im * 255.0).astype(np.uint8)
    show_in_im2 = Image.fromarray(show_in_im1)

    if 'rand' in return_what:
        x = int(np.round(np.arcsin(np.random.uniform(-1.0, 1.0)) * 160.0 / np.pi))
        y = int(np.round(np.random.uniform(-150.0, 150.0)))
        z = 0.0

    cam_vis.polar_change(x)
    cam_vis.azimuth_change(y)
    cam_vis.radius_change(z)
    cam_vis.encode_image(show_in_im1)
    new_fig = cam_vis.update_figure()

    if 'vis' in return_what:
        description = ('The viewpoints are visualized on the top right. '
                       'Click Run Generation to update the results on the bottom right.')

        if 'angles' in return_what:
            return (x, y, z, description, new_fig, show_in_im2)
        else:
            return (description, new_fig, show_in_im2)

    elif 'gen' in return_what:
        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
        input_im = input_im * 2 - 1
        input_im = transforms.functional.resize(input_im, [h, w])

        sampler = DDIMSampler(models['turncam'])
        used_x = x
        x_samples_ddim = sample_model_controlnet(input_im, control_im, models['turncam'], sampler, precision, h, w,
                                                ddim_steps, n_samples, scale, ddim_eta, used_x, y, z)

        output_ims = []
        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

        description = None

        if 'angles' in return_what:
            return (x, y, z, description, new_fig, show_in_im2, output_ims)
        else:
            return (description, new_fig, show_in_im2, output_ims)


def calc_cam_cone_pts_3d(polar_deg, azimuth_deg, radius_m, fov_deg):
    '''
    :param polar_deg (float).
    :param azimuth_deg (float).
    :param radius_m (float).
    :param fov_deg (float).
    :return (5, 3) array of float with (x, y, z).
    '''
    polar_rad = np.deg2rad(polar_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    fov_rad = np.deg2rad(fov_deg)
    polar_rad = -polar_rad

    cam_x = radius_m * np.cos(azimuth_rad) * np.cos(polar_rad)
    cam_y = radius_m * np.sin(azimuth_rad) * np.cos(polar_rad)
    cam_z = radius_m * np.sin(polar_rad)

    camera_R = np.array([[np.cos(azimuth_rad) * np.cos(polar_rad),
                          -np.sin(azimuth_rad),
                          -np.cos(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(azimuth_rad) * np.cos(polar_rad),
                          np.cos(azimuth_rad),
                          -np.sin(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(polar_rad),
                          0.0,
                          np.cos(polar_rad)]])

    corn1 = [-1.0, np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn2 = [-1.0, -np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn3 = [-1.0, -np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn4 = [-1.0, np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn1 = np.dot(camera_R, corn1)
    corn2 = np.dot(camera_R, corn2)
    corn3 = np.dot(camera_R, corn3)
    corn4 = np.dot(camera_R, corn4)

    corn1 = np.array(corn1) / np.linalg.norm(corn1, ord=2)
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    corn2 = np.array(corn2) / np.linalg.norm(corn2, ord=2)
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    corn3 = np.array(corn3) / np.linalg.norm(corn3, ord=2)
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1]
    corn_z3 = cam_z + corn3[2]
    corn4 = np.array(corn4) / np.linalg.norm(corn4, ord=2)
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4]

    return np.array([xs, ys, zs]).T


def run_demo(
        device_idx=_GPU_INDEX,
        ckpt='105000.ckpt',
        config='configs/zero123_controlnet_training.yaml'):

    print('sys.argv:', sys.argv)
    if len(sys.argv) > 1:
        print('old device_idx:', device_idx)
        try:
            device_idx = int(sys.argv[1])
            print('new device_idx:', device_idx)
        except ValueError:
            print('sys.argv[1] is not a valid device index, using default:', device_idx)

    device = f'cuda:{device_idx}'
    config = OmegaConf.load(config)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating ControlLDM...')
    models['turncam'] = load_model_from_config(config, ckpt, device=device)
    print('Instantiating Carvekit HiInterface...')
    models['carvekit'] = create_carvekit_interface()
    print('Instantiating StableDiffusionSafetyChecker...')
    models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(
        'CompVis/stable-diffusion-safety-checker').to(device)
    print('Instantiating AutoFeatureExtractor...')
    models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
        'CompVis/stable-diffusion-safety-checker')

    models['nsfw'].concept_embeds_weights *= 1.07
    models['nsfw'].special_care_embeds_weights *= 1.07

    with open('instructions.md', 'r') as f:
        article = f.read()

    # Compose demo layout & data flow.
    demo = gr.Blocks(title=_TITLE)

    with demo:
        gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=0.9, variant='panel'):

                image_block = gr.Image(type='pil', image_mode='RGBA',
                                       label='Input image of single object')
                control_image_block = gr.Image(type='pil', image_mode='L',
                                               label='Control image (edge map) - optional, will be auto-generated if not provided')
                preprocess_chk = gr.Checkbox(
                    True, label='Preprocess image automatically (remove background and recenter object)')

                gr.Markdown('*Try camera position presets:*')
                with gr.Row():
                    left_btn = gr.Button('View from the Left', variant='primary')
                    above_btn = gr.Button('View from Above', variant='primary')
                    right_btn = gr.Button('View from the Right', variant='primary')
                with gr.Row():
                    random_btn = gr.Button('Random Rotation', variant='primary')
                    below_btn = gr.Button('View from Below', variant='primary')
                    behind_btn = gr.Button('View from Behind', variant='primary')

                gr.Markdown('*Control camera position manually:*')
                polar_slider = gr.Slider(
                    -90, 90, value=0, step=5, label='Polar angle (vertical rotation in degrees)')
                azimuth_slider = gr.Slider(
                    -180, 180, value=0, step=5, label='Azimuth angle (horizontal rotation in degrees)')
                radius_slider = gr.Slider(
                    -0.5, 0.5, value=0.0, step=0.1, label='Zoom (relative distance from center)')

                samples_slider = gr.Slider(1, 8, value=4, step=1,
                                           label='Number of samples to generate')

                with gr.Accordion('Advanced options', open=False):
                    scale_slider = gr.Slider(0, 30, value=3, step=1,
                                             label='Diffusion guidance scale')
                    steps_slider = gr.Slider(5, 200, value=75, step=5,
                                             label='Number of diffusion inference steps')

                with gr.Row():
                    vis_btn = gr.Button('Visualize Angles', variant='secondary')
                    run_btn = gr.Button('Run Generation', variant='primary')

                desc_output = gr.Markdown('The results will appear on the right.', visible=_SHOW_DESC)

            with gr.Column(scale=1.1, variant='panel'):

                vis_output = gr.Plot(
                    label='Relationship between input (green) and output (blue) camera poses')

                gen_output = gr.Gallery(label='Generated images from specified new viewpoint')

                preproc_output = gr.Image(type='pil', image_mode='RGB',
                                          label='Preprocessed input image', visible=_SHOW_INTERMEDIATE)

        gr.Markdown(article)

        cam_vis = CameraVisualizer(vis_output)

        vis_btn.click(fn=partial(main_run_controlnet, models, device, cam_vis, 'vis'),
                      inputs=[polar_slider, azimuth_slider, radius_slider,
                              image_block, control_image_block, preprocess_chk],
                      outputs=[desc_output, vis_output, preproc_output])

        run_btn.click(fn=partial(main_run_controlnet, models, device, cam_vis, 'gen'),
                      inputs=[polar_slider, azimuth_slider, radius_slider,
                              image_block, control_image_block, preprocess_chk,
                              scale_slider, samples_slider, steps_slider],
                      outputs=[desc_output, vis_output, preproc_output, gen_output])

        preset_inputs = [image_block, control_image_block, preprocess_chk,
                         scale_slider, samples_slider, steps_slider]
        preset_outputs = [polar_slider, azimuth_slider, radius_slider,
                          desc_output, vis_output, preproc_output, gen_output]
        left_btn.click(fn=partial(main_run_controlnet, models, device, cam_vis, 'angles_gen',
                                  0.0, -90.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        above_btn.click(fn=partial(main_run_controlnet, models, device, cam_vis, 'angles_gen',
                                   -90.0, 0.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        right_btn.click(fn=partial(main_run_controlnet, models, device, cam_vis, 'angles_gen',
                                   0.0, 90.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        random_btn.click(fn=partial(main_run_controlnet, models, device, cam_vis, 'rand_angles_gen',
                                    -1.0, -1.0, -1.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        below_btn.click(fn=partial(main_run_controlnet, models, device, cam_vis, 'angles_gen',
                                   90.0, 0.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        behind_btn.click(fn=partial(main_run_controlnet, models, device, cam_vis, 'angles_gen',
                                    0.0, 180.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)

    # Create a custom temp directory in the current working directory
    import os
    temp_dir = os.path.join(os.getcwd(), 'gradio_temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set environment variable for Gradio temp directory
    os.environ['GRADIO_TEMP_DIR'] = temp_dir
    
    demo.launch(share=True)


@torch.no_grad()
def run_inference_on_folder(
        input_folder,
        output_folder,
        ckpt='105000.ckpt',
        config='configs/zero123_controlnet_training.yaml',
        device='cuda:0',
        x=0.0, y=30.0, z=0.0,
        scale=3.0, n_samples=1, ddim_steps=50, ddim_eta=1.0,
        h=256, w=256):

    import os
    input_folder = os.path.abspath(input_folder)
    folder_name = os.path.basename(input_folder.rstrip("/"))
    output_folder = os.path.abspath(output_folder)  # Make sure output_folder is absolute
    output_folder = os.path.join(output_folder, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Input folder: {input_folder}")
    print(f"Output folder created: {output_folder}")

    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, device)
    model.eval()
    sampler = DDIMSampler(model)

    from torchvision import transforms
    from PIL import Image

    img_paths = sorted([p for p in os.listdir(input_folder) if p.endswith('.png')])

    for img_path in img_paths:
        print(f'Processing: {img_path}')
        raw_img = Image.open(os.path.join(input_folder, img_path)).convert('RGBA')
        raw_img = raw_img.resize([256, 256], Image.Resampling.LANCZOS)
        
        # Try to load existing control image first
        control_img_path = os.path.join(input_folder, 'control', f"{img_path[:-4]}_control.png")
        if os.path.exists(control_img_path):
            print(f"Loading existing control image: {control_img_path}")
            control_im = Image.open(control_img_path).convert('L')
            # Resize control image to match input size
            control_im = control_im.resize([256, 256], Image.Resampling.LANCZOS)
        else:
            print(f"Creating new control image for {img_path}")
            # Create control image (edge map) from the original image
            control_im = create_edge_control(raw_img)
            if control_im is None:
                print(f"Failed to create control image for {img_path}")
                continue
            
        # Process input image for the model
        input_im = np.asarray(raw_img, dtype=np.float32) / 255.0

        alpha = input_im[:, :, 3:4]
        white = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white
        input_im = input_im[:, :, :3]
        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
        input_im = transforms.functional.resize(input_im * 2 - 1, [h, w])

        samples = sample_model_controlnet(input_im, control_im, model, sampler, 'fp32', h, w,
                                         ddim_steps, n_samples, scale, ddim_eta,
                                         x, y, z)
        for i, x_sample in enumerate(samples):
            out_img = 255. * rearrange(x_sample, 'c h w -> h w c').cpu().numpy()
            out_img = Image.fromarray(out_img.astype(np.uint8))
            out_img.save(os.path.join(output_folder, f'{img_path[:-4]}_pred_{i}.png'))


if __name__ == '__main__':
    import fire
    fire.Fire({
        'gradio': run_demo,
        'infer': run_inference_on_folder,
    })
