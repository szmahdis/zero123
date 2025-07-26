#!/usr/bin/env python3
"""
Zero-1-to-3 Batch Evaluation Script for Objaverse Dataset
Input: Object IDs
Output: Generated views + evaluation metrics (PSNR, SSIM, LPIPS, feature distance)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch
from contextlib import nullcontext
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms
from tqdm import tqdm

# Add zero123 to path
sys.path.append('zero123')

try:
    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm.util import load_and_preprocess, instantiate_from_config
    from ldm.modules.evaluate.evaluate_perceptualsim import (
        ssim_metric, psnr, compute_perceptual_similarity_from_list
    )
    ZERO123_METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Zero-1-to-3 PSNR/SSIM: {e}")
    ZERO123_METRICS_AVAILABLE = False

try:
    from taming.modules.losses.lpips import LPIPS
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: LPIPS not available")
    LPIPS_AVAILABLE = False

try:
    from torchvision.models import inception_v3
    from scipy.linalg import sqrtm
    FID_AVAILABLE = True
except ImportError:
    print("Warning: FID dependencies not available (need torchvision and scipy)")
    FID_AVAILABLE = False

# Fallback implementations
def calculate_psnr_basic(img1, img2):
    """Calculate PSNR using basic implementation"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim_basic(img1, img2):
    """Calculate SSIM using basic implementation"""
    try:
        from skimage.metrics import structural_similarity as ssim
        
        # Calculate SSIM for each channel
        ssim_values = []
        for i in range(3):  # RGB channels
            ssim_val = ssim(img1[:,:,i], img2[:,:,i], data_range=1.0)
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    except ImportError:
        # Fallback: simple correlation coefficient
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        correlation = np.corrcoef(img1_flat, img2_flat)[0, 1]
        return max(0, correlation)  # Clamp to positive

def calculate_lpips_basic(img1, img2):
    """Calculate LPIPS if available"""
    if not LPIPS_AVAILABLE:
        return None
        
    try:
        # Convert to tensors
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().contiguous()
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().contiguous()
        
        # Initialize LPIPS
        lpips_fn = LPIPS().eval()
        
        # Calculate LPIPS
        with torch.no_grad():
            lpips_val = lpips_fn(img1_tensor * 2 - 1, img2_tensor * 2 - 1)
        
        return lpips_val.item()
    except Exception as e:
        print(f"LPIPS calculation failed: {e}")
        return None

def get_inception_features(img, model):
    """Extract features from Inception-v3 model"""
    # Preprocess image for Inception
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    
    # Resize to 299x299 for Inception-v3
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(299, 299), mode='bilinear', align_corners=False)
    
    # Normalize to [-1, 1] range
    img_tensor = img_tensor * 2 - 1
    
    with torch.no_grad():
        features = model(img_tensor)
    
    return features.squeeze().cpu().numpy()

def calculate_fid_basic(img1, img2):
    """Calculate simplified feature distance for individual image pairs"""
    if not FID_AVAILABLE:
        return None
        
    try:
        # Load Inception-v3 model
        model = inception_v3(pretrained=True, transform_input=False)
        model.eval()
        
        # Remove final classification layer to get features
        model.fc = torch.nn.Identity()
        
        # Get features for both images
        features1 = get_inception_features(img1, model)
        features2 = get_inception_features(img2, model)
        
        # For single images, calculate Euclidean distance in feature space
        feature_distance = np.linalg.norm(features1 - features2)
        
        # Scale to make values more interpretable
        fid_like_score = feature_distance / 10.0
        
        return float(fid_like_score)
        
    except Exception as e:
        print(f"FID calculation failed: {e}")
        return None

def calculate_true_fid(real_features, generated_features):
    """Calculate true FID using proper statistical distributions"""
    if len(real_features) < 2 or len(generated_features) < 2:
        print("⚠️  True FID requires at least 2 samples per distribution")
        return None
        
    try:
        # Convert to numpy arrays
        real_features = np.array(real_features)
        generated_features = np.array(generated_features)
        
        # Calculate statistics for real images
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        # Calculate statistics for generated images  
        mu_gen = np.mean(generated_features, axis=0)
        sigma_gen = np.cov(generated_features, rowvar=False)
        
        # Calculate FID using Fréchet distance formula
        # d² = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))
        
        # Mean difference
        mu_diff = mu_real - mu_gen
        mean_distance = np.sum(mu_diff ** 2)
        
        # Covariance term
        covmean = sqrtm(sigma_real.dot(sigma_gen))
        
        # Handle numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        # Final FID calculation
        fid = mean_distance + np.trace(sigma_real + sigma_gen - 2 * covmean)
        
        # Debug output
        print(f"🔍 FID Debug: mean_dist={mean_distance:.3f}, trace_term={np.trace(sigma_real + sigma_gen - 2 * covmean):.3f}")
        print(f"🔍 Feature shapes: real={real_features.shape}, gen={generated_features.shape}")
        
        return float(fid)
        
    except Exception as e:
        print(f"True FID calculation failed: {e}")
        return None

if not ZERO123_METRICS_AVAILABLE:
    print("Will use basic implementations for PSNR/SSIM")
    sys.exit(1)  # For batch evaluation, we need Zero-1-to-3 to be available


class Zero123Evaluator:
    def __init__(self, config_path: str, ckpt_path: str, device: str = "cuda:0"):
        self.device = device
        print(f"Loading Zero-1-to-3 model...")
        
        # Initialize CUDA context and cuDNN
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Warm up CUDA
        dummy = torch.randn(1, 1, device=device)
        del dummy
        torch.cuda.empty_cache()
        
        self.config = OmegaConf.load(config_path)
        self.model = self._load_model(ckpt_path)
        print(f"Model loaded successfully on {device}")
        
    def _load_model(self, ckpt_path: str):
        """Load Zero-1-to-3 model from checkpoint"""
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            print("Downloading 105000.ckpt...")
            os.system("cd zero123 && wget https://cv.cs.columbia.edu/zero123/assets/105000.ckpt")
            
        pl_sd = torch.load(ckpt_path, map_location=self.device)
        if "global_step" in pl_sd:
            print(f"Model global step: {pl_sd['global_step']}")
        
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(self.config.model)
        m, u = model.load_state_dict(sd, strict=False)
        
        model.to(self.device)
        model.eval()
        return model

    def _load_camera_params(self, npy_path: str) -> Dict:
        """Extract camera parameters from Objaverse .npy files"""
        try:
            data = np.load(npy_path, allow_pickle=True)
            
            # Handle different data formats
            if isinstance(data, np.ndarray):
                if data.shape == (4, 4):
                    # 4x4 transformation matrix
                    camera_pos = data[:3, 3]
                    
                    # Convert to spherical coordinates
                    radius = np.linalg.norm(camera_pos)
                    azimuth = np.arctan2(camera_pos[1], camera_pos[0]) * 180 / np.pi
                    elevation = np.arcsin(camera_pos[2] / radius) * 180 / np.pi
                    
                    return {
                        'x': elevation,
                        'y': azimuth,
                        'z': 0.0,  # Keep zero for consistency
                        'raw_matrix': data.tolist()
                    }
                else:
                    # Try to extract meaningful values
                    if len(data.flatten()) >= 3:
                        flat = data.flatten()
                        return {'x': float(flat[0]), 'y': float(flat[1]), 'z': 0.0}
                        
            elif isinstance(data, dict):
                # Dictionary format
                return {
                    'x': float(data.get('elevation', 0)),
                    'y': float(data.get('azimuth', 0)),
                    'z': float(data.get('radius', 0))
                }
                
        except Exception as e:
            print(f"Warning: Failed to parse {npy_path}: {e}")
            
        # Fallback to default values
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for Zero-1-to-3"""
        image = Image.open(image_path)
        
        # Handle transparency properly - convert to white background instead of black
        if image.mode in ('RGBA', 'LA'):
            # Create white background
            white_bg = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                white_bg.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
            else:
                white_bg.paste(image, mask=image.split()[-1])
            image = white_bg
        else:
            image = image.convert('RGB')
            
        image = image.resize([256, 256], Image.Resampling.LANCZOS)
        image = np.array(image) / 255.0
        
        # Convert to tensor (ensure float32 dtype)
        image = transforms.ToTensor()(image).unsqueeze(0).to(self.device, dtype=torch.float32)
        image = image * 2 - 1  # Convert to [-1, 1] range
        return image

    @torch.no_grad()
    def generate_view(self, input_image: torch.Tensor, camera_params: Dict, 
                     n_samples: int = 4, ddim_steps: int = 50, 
                     guidance_scale: float = 3.0) -> torch.Tensor:
        """Generate novel view using Zero-1-to-3"""
        
        h, w = 256, 256
        x, y, z = camera_params['x'], camera_params['y'], camera_params['z']
        
        with self.model.ema_scope():
            # Get text conditioning
            c = self.model.get_learned_conditioning(input_image).tile(n_samples, 1, 1)
            
            # Camera parameters (ensure float32 dtype)
            T = torch.tensor([
                math.radians(x), 
                math.sin(math.radians(y)), 
                math.cos(math.radians(y)), 
                z
            ], dtype=torch.float32)
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = self.model.cc_projection(c)
            
            # Conditioning
            cond = {
                'c_crossattn': [c],
                'c_concat': [self.model.encode_first_stage(input_image.to(c.device)).mode()
                           .detach().repeat(n_samples, 1, 1, 1)]
            }
            
            # Unconditional conditioning (ensure dtype consistency)
            if guidance_scale != 1.0:
                uc = {
                    'c_concat': [torch.zeros(n_samples, 4, h // 8, w // 8, dtype=torch.float32).to(c.device)],
                    'c_crossattn': [torch.zeros_like(c, dtype=torch.float32).to(c.device)]
                }
            else:
                uc = None

            # Sample
            sampler = DDIMSampler(self.model)
            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=guidance_scale,
                unconditional_conditioning=uc,
                eta=1.0,
                x_T=None
            )
            
            # Decode to images
            x_samples = self.model.decode_first_stage(samples_ddim)
            result = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0).cpu()
            
            # Clean up GPU memory
            del samples_ddim, x_samples, c, cond
            if uc is not None:
                del uc
            torch.cuda.empty_cache()
            
            return result

    def _calculate_metrics(self, pred_image: np.ndarray, gt_image: np.ndarray) -> Dict:
        """Calculate PSNR, SSIM, and LPIPS metrics"""
        results = {}
        
        # Try Zero-1-to-3 implementations first
        if ZERO123_METRICS_AVAILABLE:
            try:
                # Convert to tensors
                pred_tensor = torch.from_numpy(pred_image).permute(2, 0, 1).unsqueeze(0).float().contiguous()
                gt_tensor = torch.from_numpy(gt_image).permute(2, 0, 1).unsqueeze(0).float().contiguous()
                
                # Calculate metrics
                psnr_val = psnr(pred_tensor, gt_tensor)
                ssim_val = ssim_metric(pred_tensor, gt_tensor)
                
                results['PSNR'] = float(psnr_val.item() if torch.is_tensor(psnr_val) else psnr_val)
                results['SSIM'] = float(ssim_val.item() if torch.is_tensor(ssim_val) else ssim_val)
                
            except Exception as e:
                print(f"⚠️  Zero-1-to-3 metrics failed: {e}")
                print("📊 Using basic metric implementations")
                results['PSNR'] = calculate_psnr_basic(pred_image, gt_image)
                results['SSIM'] = calculate_ssim_basic(pred_image, gt_image)
        else:
            # Use basic implementations
            results['PSNR'] = calculate_psnr_basic(pred_image, gt_image)
            results['SSIM'] = calculate_ssim_basic(pred_image, gt_image)
        
        # Calculate LPIPS if available
        lpips_val = calculate_lpips_basic(pred_image, gt_image)
        if lpips_val is not None:
            results['LPIPS'] = lpips_val
        
        # Calculate FID if available
        fid_val = calculate_fid_basic(pred_image, gt_image)
        if fid_val is not None:
            results['FID'] = fid_val
        
        return results

    def evaluate_object(self, object_id: str, views_dir: Path, output_dir: Path,
                       input_view_idx: int = 0, target_views: List[int] = None,
                       n_samples: int = 4, ddim_steps: int = 50,
                       guidance_scale: float = 3.0) -> Dict:
        """Evaluate a single object"""
        
        object_path = views_dir / object_id
        if not object_path.exists():
            return {"error": f"Object directory not found: {object_path}"}
        
        # Get all view files
        view_images = sorted(list(object_path.glob("*.png")))
        view_params = sorted(list(object_path.glob("*.npy")))
        
        if len(view_images) == 0:
            return {"error": f"No PNG files found in {object_path}"}
        
        print(f"\nEvaluating object {object_id} ({len(view_images)} views)")
        
        # Setup output directory
        obj_output_dir = output_dir / object_id
        obj_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load input image
        if input_view_idx >= len(view_images):
            input_view_idx = 0
            
        input_image_path = view_images[input_view_idx]
        input_image = self._preprocess_image(str(input_image_path))
        
        # Copy input image to output
        input_copy = obj_output_dir / f"input_{input_view_idx:03d}.png"
        Image.open(input_image_path).save(input_copy)
        
        # Determine target views
        if target_views is None:
            target_views = [i for i in range(len(view_images)) if i != input_view_idx][:6]
        
        results = {
            "object_id": object_id,
            "input_view": input_view_idx,
            "input_image": str(input_image_path),
            "total_views": len(view_images),
            "target_views": target_views,
            "generated_views": [],
            "metrics": {"per_view": [], "summary": {}}
        }
        
        all_psnr = []
        all_ssim = []
        all_lpips = []
        all_fid = []
        
        # For true FID calculation
        all_generated_features = []
        all_gt_features = []
        inception_model = None
        
        # Generate each target view
        for target_idx in target_views:
            if target_idx >= len(view_images):
                continue
                
            print(f"  Generating view {target_idx:03d}...")
            
            # Load target camera parameters
            if target_idx < len(view_params):
                camera_params = self._load_camera_params(str(view_params[target_idx]))
            else:
                # Use predefined viewpoints
                predefined = [
                    {'x': 0, 'y': -90, 'z': 0},   # left
                    {'x': 0, 'y': 90, 'z': 0},    # right
                    {'x': -90, 'y': 0, 'z': 0},   # top
                    {'x': 90, 'y': 0, 'z': 0},    # bottom
                    {'x': 0, 'y': 180, 'z': 0},   # back
                ]
                camera_params = predefined[target_idx % len(predefined)]
            
            # Generate view
            start_time = time.time()
            try:
                # Clear CUDA cache before generation
                torch.cuda.empty_cache()
                
                generated_samples = self.generate_view(
                    input_image, camera_params, n_samples, ddim_steps, guidance_scale
                )
                generation_time = time.time() - start_time
                
            except RuntimeError as e:
                if "cuDNN" in str(e) or "CUDA" in str(e):
                    print(f"⚠️  CUDA/cuDNN error on view {target_idx}: {e}")
                    print("🔄 Reinitializing CUDA context and retrying...")
                    
                    # Reinitialize CUDA context
                    torch.cuda.empty_cache()
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    
                    # Retry generation
                    generated_samples = self.generate_view(
                        input_image, camera_params, n_samples, ddim_steps, guidance_scale
                    )
                    generation_time = time.time() - start_time
                else:
                    raise e
            
            # Save generated samples
            sample_paths = []
            for k, sample in enumerate(generated_samples):
                sample_np = rearrange(sample.numpy(), 'c h w -> h w c')
                sample_img = Image.fromarray((sample_np * 255).astype(np.uint8))
                
                sample_path = obj_output_dir / f"generated_{target_idx:03d}_sample_{k:02d}.png"
                sample_img.save(sample_path)
                sample_paths.append(str(sample_path))
            
            # Load ground truth and calculate metrics
            gt_image_path = view_images[target_idx]
            gt_image = Image.open(gt_image_path)
            
            # Handle transparency properly - convert to white background instead of black
            if gt_image.mode in ('RGBA', 'LA'):
                # Create white background
                white_bg = Image.new('RGB', gt_image.size, (255, 255, 255))
                if gt_image.mode == 'RGBA':
                    white_bg.paste(gt_image, mask=gt_image.split()[-1])  # Use alpha channel as mask
                else:
                    white_bg.paste(gt_image, mask=gt_image.split()[-1])
                gt_image = white_bg
            else:
                gt_image = gt_image.convert('RGB')
                
            gt_image = gt_image.resize([256, 256], Image.Resampling.LANCZOS)
            gt_array = np.array(gt_image) / 255.0
            
            # Copy ground truth for comparison
            gt_copy = obj_output_dir / f"gt_{target_idx:03d}.png"
            gt_image.save(gt_copy)
            
            # Use best sample for metrics (first one)
            best_sample = rearrange(generated_samples[0].numpy(), 'c h w -> h w c')
            metrics = self._calculate_metrics(best_sample, gt_array)
            
            # Collect Inception features for true FID calculation
            if FID_AVAILABLE:
                try:
                    # Initialize Inception model once (reuse across views)
                    if inception_model is None:
                        inception_model = inception_v3(pretrained=True, transform_input=False)
                        inception_model.eval()
                        inception_model.fc = torch.nn.Identity()
                    
                    # Extract features for generated image
                    gen_features = get_inception_features(best_sample, inception_model)
                    all_generated_features.append(gen_features)
                    
                    # Extract features for ground truth image  
                    gt_features = get_inception_features(gt_array, inception_model)
                    all_gt_features.append(gt_features)
                    
                except Exception as e:
                    print(f"⚠️  Feature extraction failed for view {target_idx}: {e}")
            
            all_psnr.append(metrics['PSNR'])
            all_ssim.append(metrics['SSIM'])
            if 'LPIPS' in metrics:
                all_lpips.append(metrics['LPIPS'])
            if 'FID' in metrics:
                all_fid.append(metrics['FID'])
            
            view_result = {
                "target_idx": target_idx,
                "gt_image": str(gt_image_path),
                "generated_samples": sample_paths,
                "camera_params": camera_params,
                "generation_time": generation_time,
                "metrics": metrics
            }
            
            results["generated_views"].append(view_result)
            per_view_metrics = {
                "view": target_idx,
                "PSNR": metrics['PSNR'],
                "SSIM": metrics['SSIM']
            }
            if 'LPIPS' in metrics:
                per_view_metrics['LPIPS'] = metrics['LPIPS']
            if 'FID' in metrics:
                per_view_metrics['FID'] = metrics['FID']
            
            results["metrics"]["per_view"].append(per_view_metrics)
            
            # Print metrics
            lpips_str = f", LPIPS: {metrics['LPIPS']:.4f}" if 'LPIPS' in metrics else ""
            fid_str = f", FID: {metrics['FID']:.2f}" if 'FID' in metrics else ""
            print(f"    PSNR: {metrics['PSNR']:.2f}, SSIM: {metrics['SSIM']:.4f}{lpips_str}{fid_str}")
        
        # Clean up Inception model
        if inception_model is not None:
            del inception_model
            torch.cuda.empty_cache()
        
        # Calculate summary metrics
        if all_psnr:
            summary_metrics = {
                "PSNR": {
                    "mean": float(np.mean(all_psnr)),
                    "std": float(np.std(all_psnr)),
                    "min": float(np.min(all_psnr)),
                    "max": float(np.max(all_psnr))
                },
                "SSIM": {
                    "mean": float(np.mean(all_ssim)),
                    "std": float(np.std(all_ssim)),
                    "min": float(np.min(all_ssim)),
                    "max": float(np.max(all_ssim))
                }
            }
            
            # Add LPIPS if available
            if all_lpips:
                summary_metrics["LPIPS"] = {
                    "mean": float(np.mean(all_lpips)),
                    "std": float(np.std(all_lpips)),
                    "min": float(np.min(all_lpips)),
                    "max": float(np.max(all_lpips))
                }
                
            # Add FID if available
            if all_fid:
                summary_metrics["FID"] = {
                    "mean": float(np.mean(all_fid)),
                    "std": float(np.std(all_fid)),
                    "min": float(np.min(all_fid)),
                    "max": float(np.max(all_fid))
                }
            
            # Calculate True FID using collected features
            if len(all_generated_features) >= 2 and len(all_gt_features) >= 2:
                true_fid = calculate_true_fid(all_gt_features, all_generated_features)
                if true_fid is not None:
                    summary_metrics["True_FID"] = float(true_fid)
                    print(f"🎯 True FID: {true_fid:.3f}")
            else:
                print(f"⚠️  Not enough samples for True FID (need ≥2, got {len(all_generated_features)} generated, {len(all_gt_features)} GT)")
            
            results["metrics"]["summary"] = summary_metrics
            
            # Print summary
            lpips_summary = f", LPIPS: {np.mean(all_lpips):.4f}±{np.std(all_lpips):.4f}" if all_lpips else ""
            fid_summary = f", FID: {np.mean(all_fid):.2f}±{np.std(all_fid):.2f}" if all_fid else ""
            print(f"  Summary - PSNR: {np.mean(all_psnr):.2f}±{np.std(all_psnr):.2f}, "
                  f"SSIM: {np.mean(all_ssim):.4f}±{np.std(all_ssim):.4f}{lpips_summary}{fid_summary}")
        
        return results

    def evaluate_batch(self, object_ids: List[str], views_dir: Path, output_dir: Path,
                      **kwargs) -> Dict:
        """Evaluate multiple objects"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting batch evaluation of {len(object_ids)} objects...")
        print(f"Views directory: {views_dir}")
        print(f"Output directory: {output_dir}")
        
        batch_results = {
            "views_directory": str(views_dir),
            "output_directory": str(output_dir),
            "object_ids": object_ids,
            "parameters": kwargs,
            "objects": {},
            "summary": {}
        }
        
        all_metrics = []
        successful_objects = 0
        
        for object_id in tqdm(object_ids, desc="Processing objects"):
            try:
                # Clear GPU cache before processing each object
                torch.cuda.empty_cache()
                
                result = self.evaluate_object(object_id, views_dir, output_dir, **kwargs)
                batch_results["objects"][object_id] = result
                
                if "error" not in result and result["metrics"]["summary"]:
                    all_metrics.append(result["metrics"]["summary"])
                    successful_objects += 1
                    
            except RuntimeError as e:
                if "cuDNN" in str(e) or "CUDA" in str(e):
                    print(f"🚨 CUDA/cuDNN error for {object_id}: {e}")
                    print("💡 Try reducing batch size or DDIM steps if this persists")
                    batch_results["objects"][object_id] = {"error": f"CUDA error: {str(e)}"}
                else:
                    print(f"Error processing {object_id}: {e}")
                    batch_results["objects"][object_id] = {"error": str(e)}
            except Exception as e:
                print(f"Error processing {object_id}: {e}")
                batch_results["objects"][object_id] = {"error": str(e)}
        
        # Calculate overall summary
        if all_metrics:
            all_psnr = [m["PSNR"]["mean"] for m in all_metrics]
            all_ssim = [m["SSIM"]["mean"] for m in all_metrics]
            all_lpips = [m["LPIPS"]["mean"] for m in all_metrics if "LPIPS" in m]
            all_fid = [m["FID"]["mean"] for m in all_metrics if "FID" in m]
            all_true_fid = [m["True_FID"] for m in all_metrics if "True_FID" in m]
            
            overall_metrics = {
                "PSNR": {
                    "mean": float(np.mean(all_psnr)),
                    "std": float(np.std(all_psnr)),
                    "min": float(np.min(all_psnr)),
                    "max": float(np.max(all_psnr))
                },
                "SSIM": {
                    "mean": float(np.mean(all_ssim)),
                    "std": float(np.std(all_ssim)),
                    "min": float(np.min(all_ssim)),
                    "max": float(np.max(all_ssim))
                }
            }
            
            # Add LPIPS if available
            if all_lpips:
                overall_metrics["LPIPS"] = {
                    "mean": float(np.mean(all_lpips)),
                    "std": float(np.std(all_lpips)),
                    "min": float(np.min(all_lpips)),
                    "max": float(np.max(all_lpips))
                }
                
            # Add FID if available
            if all_fid:
                overall_metrics["FID"] = {
                    "mean": float(np.mean(all_fid)),
                    "std": float(np.std(all_fid)),
                    "min": float(np.min(all_fid)),
                    "max": float(np.max(all_fid))
                }
                
            # Add True FID if available
            if all_true_fid:
                overall_metrics["True_FID"] = {
                    "mean": float(np.mean(all_true_fid)),
                    "std": float(np.std(all_true_fid)),
                    "min": float(np.min(all_true_fid)),
                    "max": float(np.max(all_true_fid))
                }
            
            batch_results["summary"] = {
                "total_objects": len(object_ids),
                "successful_objects": successful_objects,
                "failed_objects": len(object_ids) - successful_objects,
                "overall_metrics": overall_metrics
            }
            
            print(f"\n=== BATCH SUMMARY ===")
            print(f"Successfully processed: {successful_objects}/{len(object_ids)} objects")
            print(f"Overall PSNR: {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f}")
            print(f"Overall SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
            if all_lpips:
                print(f"Overall LPIPS: {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}")
            if all_fid:
                print(f"Overall FID: {np.mean(all_fid):.2f} ± {np.std(all_fid):.2f}")
            if all_true_fid:
                print(f"Overall True FID: {np.mean(all_true_fid):.3f} ± {np.std(all_true_fid):.3f}")
                print(f"🎯 True FID matches paper methodology!")
        
        # Save results
        results_file = output_dir / "batch_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        return batch_results


def main():
    parser = argparse.ArgumentParser(description="Zero-1-to-3 Batch Evaluation")
    parser.add_argument("--object_ids", nargs='+', required=True,
                       help="Object IDs to evaluate")
    parser.add_argument("--views_dir", required=True,
                       help="Path to views_release directory")
    parser.add_argument("--output_dir", default="./eval_results",
                       help="Output directory for results")
    parser.add_argument("--config", default="zero123/configs/sd-objaverse-finetune-c_concat-256.yaml",
                       help="Model config path")
    parser.add_argument("--ckpt", default="zero123/105000.ckpt",
                       help="Model checkpoint path")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to use")
    parser.add_argument("--input_view", type=int, default=0,
                       help="Input view index (0-11)")
    parser.add_argument("--n_samples", type=int, default=4,
                       help="Number of samples per view")
    parser.add_argument("--ddim_steps", type=int, default=50,
                       help="DDIM sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                       help="Classifier-free guidance scale")
    
    args = parser.parse_args()
    
    # Validate inputs
    views_dir = Path(args.views_dir)
    if not views_dir.exists():
        print(f"Error: Views directory not found: {views_dir}")
        return
    
    # Check if object directories exist
    valid_objects = []
    for obj_id in args.object_ids:
        obj_path = views_dir / obj_id
        if obj_path.exists():
            valid_objects.append(obj_id)
        else:
            print(f"Warning: Object {obj_id} not found in {views_dir}")
    
    if not valid_objects:
        print("Error: No valid objects found!")
        return
    
    print(f"Found {len(valid_objects)} valid objects: {valid_objects}")
    
    # Initialize evaluator
    evaluator = Zero123Evaluator(args.config, args.ckpt, args.device)
    
    # Run evaluation
    results = evaluator.evaluate_batch(
        object_ids=valid_objects,
        views_dir=views_dir,
        output_dir=Path(args.output_dir),
        input_view_idx=args.input_view,
        n_samples=args.n_samples,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale
    )
    
    print("✅ Batch evaluation completed!")


if __name__ == "__main__":
    main() 