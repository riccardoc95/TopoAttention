""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import io
import logging
from typing import Optional

import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as T
from matplotlib import pyplot as plt

from .transforms import *
from ._topo import *

from .readers import create_reader

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            input_img_mode='RGB',
            transform=None,
            target_transform=None,
            **kwargs,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map,
                **kwargs,
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.input_img_mode = input_img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.reader[index]

        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.input_img_mode and not self.load_bytes:
            img = img.convert(self.input_img_mode)
        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            is_training=False,
            batch_size=1,
            num_samples=None,
            seed=42,
            repeats=0,
            download=False,
            input_img_mode='RGB',
            input_key=None,
            target_key=None,
            transform=None,
            target_transform=None,
            max_steps=None,
    ):
        assert reader is not None
        if isinstance(reader, str):
            self.reader = create_reader(
                reader,
                root=root,
                split=split,
                class_map=class_map,
                is_training=is_training,
                batch_size=batch_size,
                num_samples=num_samples,
                seed=seed,
                repeats=repeats,
                download=download,
                input_img_mode=input_img_mode,
                input_key=input_key,
                target_key=target_key,
                max_steps=max_steps,
            )
        else:
            self.reader = reader
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.reader:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.reader, '__len__'):
            return len(self.reader)
        else:
            return 0

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self.reader, 'set_epoch'):
            self.reader.set_epoch(count)

    def set_loader_cfg(
            self,
            num_workers: Optional[int] = None,
    ):
        # TFDS and WDS readers need # workers for correct # samples estimate before loader processes created
        if hasattr(self.reader, 'set_loader_cfg'):
            self.reader.set_loader_cfg(num_workers=num_workers)

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)



# Function to check if a transformation is in the pipeline
def has_transform(transform_pipeline, transform_type):
    return any(isinstance(t, transform_type) for t in transform_pipeline.transforms)


# Function to remove a specific transformation
def remove_transform(transform_pipeline, transform_type):
    new_transforms = [t for t in transform_pipeline.transforms if not isinstance(t, transform_type)]
    return transforms.Compose(new_transforms)


class DenoisingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, noise_level=0.1):
        """
        Custom Dataset for image denoising.

        Args:
            root (str): Path to the dataset folder (compatible with torchvision.datasets.ImageFolder).
            transform (callable, optional): Transform to apply to the clean images.
            noise_level (float, optional): Standard deviation of Gaussian noise to add (default: 0.1).
        """
        self.dataset = dataset
        self.transform = transform
        self.noise_level = noise_level
        self.dataset.transform = None
        self.dataset.target_transform = None

    def add_noise(self, image):
        """
        Adds Gaussian noise to the image.

        Args:
            image (Tensor): Clean image tensor.

        Returns:
            Tensor: Noisy image tensor.
        """
        noise = np.random.randn(*image.shape) * self.noise_level
        noised_image = image + noise
        return np.clip(noised_image, 0.0, 1.0)  # Ensure pixel values remain in [0, 1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Fetch a clean image and its noised counterpart.

        Args:
            idx (int): Index of the image.

        Returns:
            Tuple[Tensor, Tensor]: (Noised image, Clean image)
        """
        image = self.dataset[idx][0]  # Ignore the label from ImageFolder
        noised_image = self.dataset[idx][0].copy()

        if len(np.array(image).shape) == 3:
            image = np.array(image).transpose((2, 0, 1)).astype(np.float32) / 255.
            noised_image = np.array(noised_image).transpose((2, 0, 1)).astype(np.float32) / 255.
        else:
            image = np.array(image).astype(np.float32) / 255.
            noised_image = np.array(noised_image).astype(np.float32) / 255.
            image = np.expand_dims(image, axis=0)
            noised_image = np.expand_dims(noised_image, axis=0)

        noised_image = self.add_noise(noised_image)

        if self.transform and has_transform(self.transform, TopoTransform):
            noised_image = TopoTransform()(noised_image)


        noised_image = torch.from_numpy(noised_image).float()
        image = torch.from_numpy(image).float()

        image = (image - 0.5) / 0.5
        return noised_image, image


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, segmentation_level=0.5):
        """
        Custom Dataset for image denoising.

        Args:
            root (str): Path to the dataset folder (compatible with torchvision.datasets.ImageFolder).
            transform (callable, optional): Transform to apply to the clean images.
            noise_level (float, optional): Standard deviation of Gaussian noise to add (default: 0.1).
        """
        self.dataset = dataset
        self.transform = transform
        self.segmentation_level = segmentation_level
        self.dataset.transform = None
        self.dataset.target_transform = None

    def __len__(self):
        return len(self.dataset)

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        mask = self.dataset[idx][0].copy()

        if len(np.array(image).shape) == 3:
            image = np.array(image).transpose((2, 0, 1)).astype(np.float32) / 255.
            mask = np.array(mask).transpose((2, 0, 1)).astype(np.float32) / 255.
        else:
            image = np.array(image).astype(np.float32) / 255.
            mask = np.array(mask).astype(np.float32) / 255.
            image = np.expand_dims(image, axis=0)
            #mask = np.expand_dims(mask, axis=0)

        if len(mask.shape) == 3:
            mask = (self.rgb2gray(mask) > self.segmentation_level) * 1
        else:
            mask = (mask > self.segmentation_level) * 1
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, axis=0)

        if self.transform and has_transform(self.transform, TopoTransform):
            image = TopoTransform()(image)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()#.long()

        return image, mask