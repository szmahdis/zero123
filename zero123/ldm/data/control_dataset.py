import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
import random


class ControlTextImageDataset(Dataset):
    def __init__(self, data_root, control_root, size=None, interpolation="bicubic", 
                 flip_p=0.5, set="train", placeholder_token="*", control_channels=3):
        """
        Dataset for ControlNet training with text-image-control triplets.
        
        Args:
            data_root: Path to the main image data
            control_root: Path to control images/feature maps
            size: Size to resize images to
            interpolation: Interpolation method for resizing
            flip_p: Probability of horizontal flip
            set: "train" or "validation"
            placeholder_token: Token to replace in captions
            control_channels: Number of channels in control images (3 for RGB, 1 for grayscale, etc.)
        """
        self.data_root = data_root
        self.control_root = control_root
        self.size = size
        self.interpolation = {"linear": Image.LINEAR,
                             "bilinear": Image.BILINEAR,
                             "bicubic": Image.BICUBIC,
                             "lanczos": Image.LANCZOS,
                             }[interpolation]
        self.flip_p = flip_p
        self.set = set
        self.placeholder_token = placeholder_token
        self.control_channels = control_channels

        # Get all image files
        self.image_paths = []
        self.control_paths = []
        self.captions = []
        
        # Walk through data directory
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    # Get relative path
                    rel_path = os.path.relpath(os.path.join(root, file), data_root)
                    
                    # Check if corresponding control image exists
                    control_path = os.path.join(control_root, rel_path)
                    if os.path.exists(control_path):
                        self.image_paths.append(os.path.join(data_root, rel_path))
                        self.control_paths.append(control_path)
                        
                        # Try to find caption file
                        caption_path = os.path.join(data_root, rel_path.rsplit('.', 1)[0] + '.txt')
                        if os.path.exists(caption_path):
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                caption = f.read().strip()
                        else:
                            # Generate a simple caption based on filename
                            caption = f"a photo of {os.path.splitext(os.path.basename(file))[0]}"
                        
                        self.captions.append(caption)

        print(f"Found {len(self.image_paths)} image-control pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        example = {}

        # Load image
        image = Image.open(self.image_paths[i]).convert("RGB")
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        # Load control image/feature map
        control = Image.open(self.control_paths[i])
        if self.control_channels == 1:
            control = control.convert("L")
        elif self.control_channels == 3:
            control = control.convert("RGB")
        
        if self.size is not None:
            control = control.resize((self.size, self.size), resample=self.interpolation)

        # Convert to numpy arrays
        image = np.array(image).astype(np.uint8)
        control = np.array(control).astype(np.uint8)

        # Apply horizontal flip with probability
        if random.random() < self.flip_p:
            image = np.flip(image, axis=1)
            control = np.flip(control, axis=1)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 127.5 - 1.0
        control = control.astype(np.float32) / 127.5 - 1.0

        # Add channel dimension if needed
        if self.control_channels == 1:
            control = control[..., None]

        example["jpg"] = image
        example["control"] = control
        example["txt"] = self.captions[i]

        return example


class ControlTextImageIterableDataset(Txt2ImgIterableBaseDataset):
    def __init__(self, data_root, control_root, size=None, interpolation="bicubic", 
                 flip_p=0.5, set="train", placeholder_token="*", control_channels=3, **kwargs):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.control_root = control_root
        self.size = size
        self.interpolation = {"linear": Image.LINEAR,
                             "bilinear": Image.BILINEAR,
                             "bicubic": Image.BICUBIC,
                             "lanczos": Image.LANCZOS,
                             }[interpolation]
        self.flip_p = flip_p
        self.set = set
        self.placeholder_token = placeholder_token
        self.control_channels = control_channels

        # Get all image files
        self.image_paths = []
        self.control_paths = []
        self.captions = []
        
        # Walk through data directory
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    # Get relative path
                    rel_path = os.path.relpath(os.path.join(root, file), data_root)
                    
                    # Check if corresponding control image exists
                    control_path = os.path.join(control_root, rel_path)
                    if os.path.exists(control_path):
                        self.image_paths.append(os.path.join(data_root, rel_path))
                        self.control_paths.append(control_path)
                        
                        # Try to find caption file
                        caption_path = os.path.join(data_root, rel_path.rsplit('.', 1)[0] + '.txt')
                        if os.path.exists(caption_path):
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                caption = f.read().strip()
                        else:
                            # Generate a simple caption based on filename
                            caption = f"a photo of {os.path.splitext(os.path.basename(file))[0]}"
                        
                        self.captions.append(caption)

        print(f"Found {len(self.image_paths)} image-control pairs")

    def __iter__(self):
        for i in range(len(self.image_paths)):
            example = {}

            # Load image
            image = Image.open(self.image_paths[i]).convert("RGB")
            if self.size is not None:
                image = image.resize((self.size, self.size), resample=self.interpolation)

            # Load control image/feature map
            control = Image.open(self.control_paths[i])
            if self.control_channels == 1:
                control = control.convert("L")
            elif self.control_channels == 3:
                control = control.convert("RGB")
            
            if self.size is not None:
                control = control.resize((self.size, self.size), resample=self.interpolation)

            # Convert to numpy arrays
            image = np.array(image).astype(np.uint8)
            control = np.array(control).astype(np.uint8)

            # Apply horizontal flip with probability
            if random.random() < self.flip_p:
                image = np.flip(image, axis=1)
                control = np.flip(control, axis=1)

            # Normalize to [0, 1]
            image = image.astype(np.float32) / 127.5 - 1.0
            control = control.astype(np.float32) / 127.5 - 1.0

            # Add channel dimension if needed
            if self.control_channels == 1:
                control = control[..., None]

            example["jpg"] = image
            example["control"] = control
            example["txt"] = self.captions[i]

            yield example


def make_control_dataset(config):
    """
    Factory function to create control datasets from config.
    """
    if config.get("iterable", False):
        return ControlTextImageIterableDataset(**config)
    else:
        return ControlTextImageDataset(**config) 