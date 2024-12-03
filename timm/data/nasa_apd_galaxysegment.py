import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import numpy as np

from ._topo import persistent_image_core, rgb2gray


class APDGalaxy(Dataset):
    def __init__(self, folder="nasa-apd-galaxysegment", dim=128, is_training=True, transform=None, mask_transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            mask_dir (str): Directory with all the segmentation masks.
            transform (callable, optional): Transformation for the images.
            mask_transform (callable, optional): Transformation for the masks.
        """
        self.image_dir = os.path.join(folder, "images")
        self.mask_dir = os.path.join(folder, "segmentation_maps")
        self.transform = transform
        self.mask_transform = mask_transform
        self.is_training = is_training
        self.dim = dim

        # Get sorted list of files for consistency
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".png")])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.image_files)

    def decode_mask(self, mask):
        colormap = {
            (0, 0, 0): 0,
            (255, 0, 0): 1,
            (0, 255, 0): 2,
            (0, 0, 255): 3,
            (255, 255, 0): 4,
        }
        mask = np.array(mask)  # Convert to numpy array
        decoded_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for color, class_value in colormap.items():
            decoded_mask[(mask == color).all(axis=-1)] = class_value
        decoded_mask = np.expand_dims(decoded_mask, axis=0)
        decoded_mask = (decoded_mask > 0).astype(np.uint8)
        return decoded_mask

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Ensure RGB format

        # Load mask
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = Image.open(mask_path).convert("RGB")  # Mask was saved as colored

        image = np.array(image).transpose(2, 0, 1)
        mask = self.decode_mask(mask)

        # Apply transformations
        #if self.transform:
        #    image = self.transform(image)
        #if self.mask_transform:
        #    mask = self.mask_transform(mask)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()#.long()

        return image, mask


class GalaxySegmentationDataset(Dataset):
    def __init__(self, folder="nasa-apd-galaxysegment", dim=128, is_training=True, transform=None, mask_transform=None, train_perc=0.8,
                 random_state=42):
        """
        Args:
            image_dir (str): Directory with all the images.
            mask_dir (str): Directory with all the segmentation masks.
            transform (callable, optional): Transformation for the images.
            mask_transform (callable, optional): Transformation for the masks.
        """
        self.dataset = APDGalaxy(folder=folder, dim=dim, is_training=is_training, transform=transform, mask_transform=mask_transform)
        self.split = "train" if is_training else "valid"
        # Define split sizes
        self.train_size = int(train_perc * len(self.dataset))
        self.valid_size = len(self.dataset) - self.train_size

        # Split dataset
        generator = torch.Generator().manual_seed(random_state)
        self.train_dataset, self.valid_dataset = random_split(self.dataset, [self.train_size, self.valid_size],
                                                              generator=generator)
        self.topo = False
    def __len__(self):
        if self.split == "train":
            return len(self.train_dataset)
        else:
            return len(self.valid_dataset)

    def __getitem__(self, idx):
        if self.split == "train":
            image, mask = self.train_dataset[idx]
        else:
            image, mask = self.valid_dataset[idx]

        if self.topo:
            topo = persistent_image_core(rgb2gray(image.numpy())[0])
            topo = topo / np.linalg.norm(topo)
            topo = torch.from_numpy(topo).unsqueeze(0).float()
            image = torch.concatenate([topo, image])
        return image, mask