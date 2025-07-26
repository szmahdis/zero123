import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torch
from einops import rearrange
from pathlib import Path
import json
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

class SimpleObjaverseData(Dataset):
    def __init__(self, root_dir, total_view=12, validation=False, image_size=256):
        self.root_dir = Path(root_dir)
        self.total_view = total_view
        self.validation = validation

        # Load valid paths
        valid_path_file = os.path.join(root_dir, 'valid_paths.json')
        with open(valid_path_file) as f:
            all_paths = json.load(f)
        print(f"Loaded {len(all_paths)} object paths from {valid_path_file}")

        # Filter out paths that don't exist or are missing required files
        self.paths = []
        for path in all_paths:
            full_path = os.path.join(self.root_dir, path)
            if not os.path.exists(full_path):
                print(f"Skipping {path}: directory does not exist ({full_path})")
                continue

            # Only consider indices with both .png and .npy
            valid_indices = []
            for i in range(self.total_view):
                png_path = os.path.join(full_path, f'{i:03d}.png')
                npy_path = os.path.join(full_path, f'{i:03d}.npy')
                control_path = os.path.join(full_path, 'control', f'{i:03d}_control.png')
                if os.path.exists(png_path) and os.path.exists(npy_path) and os.path.exists(control_path):
                    valid_indices.append(i)
            if len(valid_indices) < 2:
                print(f"Skipping {path}: not enough valid views with control images (found {len(valid_indices)}, need at least 2)")
                continue

            self.paths.append(path)

        print(f"After filtering, {len(self.paths)} objects have at least 2 views with PNG, NPY, and control images.")

        total_objects = len(self.paths)
        # Split into train/val
        if validation:
            if total_objects <= 10:
                self.paths = self.paths[-1:] if total_objects > 1 else self.paths
                print(f"Validation set: using {len(self.paths)} object(s) (last 1 if >1 objects)")
            else:
                self.paths = self.paths[math.floor(total_objects / 100. * 99.):]
                print(f"Validation set: using {len(self.paths)} object(s) (last 1%)")
        else:
            if total_objects <= 10:
                self.paths = self.paths[:-1] if total_objects > 1 else self.paths
                print(f"Training set: using {len(self.paths)} object(s) (all but last 1 if >1 objects)")
            else:
                self.paths = self.paths[:math.floor(total_objects / 100. * 99.)]
                print(f"Training set: using {len(self.paths)} object(s) (first 99%)")

        print(f'============= length of dataset {len(self.paths)} (filtered from {len(all_paths)}) =============')

        # Create transforms
        image_transforms = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
        ]
        self.image_transforms = transforms.Compose(image_transforms)
        
        # Create transforms for control images (grayscale)
        control_transforms = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: rearrange(x, 'c h w -> h w c'))  # Keep as single channel
        ]
        self.control_transforms = transforms.Compose(control_transforms)

    def __len__(self):
        return len(self.paths)

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2])
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        try:
            R, T = target_RT[:3, :3], target_RT[:3, -1]
            T_target = -R.T @ T

            R, T = cond_RT[:3, :3], cond_RT[:3, -1]
            T_cond = -R.T @ T

            theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
            theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])

            d_theta = theta_target - theta_cond
            d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
            d_z = z_target - z_cond

            d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
            return d_T
        except Exception as e:
            print(f"Error in get_T: {e}")
            return torch.tensor([0.0, 0.0, 1.0, 0.0])

    def load_im(self, path, color):
        try:
            img = plt.imread(path)
        except:
            print(f"Failed to load image: {path}")
            return Image.new('RGB', (256, 256), (0, 0, 0))
        if img.ndim == 3 and img.shape[2] == 4:
            img[img[:, :, -1] == 0.] = color
            img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        elif img.ndim == 3 and img.shape[2] == 3:
            img = Image.fromarray(np.uint8(img * 255.))
        else:
            img = Image.fromarray(np.uint8(img * 255.)).convert('RGB')
        return img

    def load_control_im(self, path):
        """Load control image (grayscale)."""
        try:
            img = plt.imread(path)
            if img.ndim == 3:
                # Convert to grayscale if RGB
                img = np.mean(img, axis=2)
            # Normalize to [0, 1]
            img = img / 255.0
            return Image.fromarray(np.uint8(img * 255.))
        except Exception as e:
            print(f"Failed to load control image: {path}, error: {e}")
            return Image.new('L', (256, 256), (0))

    def __getitem__(self, index):
        data = {}
        total_view = self.total_view
        filename = os.path.join(self.root_dir, self.paths[index])
        # Find valid indices (png, npy, and control present)
        valid_indices = [i for i in range(total_view)
                         if os.path.exists(os.path.join(filename, f'{i:03d}.png')) and
                            os.path.exists(os.path.join(filename, f'{i:03d}.npy')) and
                            os.path.exists(os.path.join(filename, 'control', f'{i:03d}_control.png'))]
        if len(valid_indices) < 2:
            raise RuntimeError(f"Object {self.paths[index]} does not have 2 valid views with control images at runtime!")

        index_target, index_cond = random.sample(valid_indices, 2)

        color = [1., 1., 1., 1.]
        try:
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            
            # Load control images
            target_control = self.process_control_im(self.load_control_im(os.path.join(filename, 'control', '%03d_control.png' % index_target)))
            cond_control = self.process_control_im(self.load_control_im(os.path.join(filename, 'control', '%03d_control.png' % index_cond)))
            
        except Exception as e:
            print(f"Error loading data from {filename}: {e}")
            target_im = torch.zeros((256, 256, 3))
            cond_im = torch.zeros((256, 256, 3))
            target_RT = np.eye(4)
            cond_RT = np.eye(4)
            target_control = torch.zeros((256, 256, 1))
            cond_control = torch.zeros((256, 256, 1))

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["control"] = cond_control  # Control image for conditioning
        data["T"] = self.get_T(target_RT, cond_RT)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.image_transforms(im)
    
    def process_control_im(self, im):
        im = im.convert("L")  # Convert to grayscale
        return self.control_transforms(im)

class SimpleObjaverseDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=2, total_view=12, num_workers=0, image_size=256):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.total_view = total_view
        self.num_workers = num_workers
        self.image_size = image_size

    def train_dataloader(self):
        dataset = SimpleObjaverseData(
            root_dir=self.root_dir, 
            total_view=self.total_view, 
            validation=False,
            image_size=self.image_size
        )
        # Only use DistributedSampler if we're actually in a distributed setting
        try:
            sampler = DistributedSampler(dataset)
            shuffle = False
        except:
            sampler = None
            shuffle = True
            
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=shuffle, 
            sampler=sampler
        )

    def val_dataloader(self):
        dataset = SimpleObjaverseData(
            root_dir=self.root_dir, 
            total_view=self.total_view, 
            validation=True,
            image_size=self.image_size
        )
        try:
            sampler = DistributedSampler(dataset)
            shuffle = False
        except:
            sampler = None
            shuffle = False
            
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=shuffle, 
            sampler=sampler
        )
