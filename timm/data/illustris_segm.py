from numpy.lib.stride_tricks import as_strided
from astropy.io import fits
import os
import numpy as np
import torch
import scipy

from ._topo import persistent_image_core

def load_illustris(folder=".", dim=128):
    hdu = fits.open(os.path.join(folder, "illustris", "illustris.fits"))
    img = hdu[3].data.astype(np.float32)
    seg = fits.getdata(os.path.join(folder, "illustris", "segmentation.fits"))
    seg = (seg > 0).astype(np.float32)

    p = np.percentile(img, 99)
    img = np.clip(img, None, p)# / p
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = 2 * img - 1
    #arr = arr / arr.max()

    window_shape = (dim, dim)
    output_shape = (img.shape[0] // window_shape[0], img.shape[1] // window_shape[1]) + window_shape
    strides = (img.strides[0] * window_shape[0], img.strides[1] * window_shape[1]) + img.strides
    data_img = as_strided(img, shape=output_shape, strides=strides).reshape(-1, dim, dim)
    data_seg = as_strided(seg, shape=output_shape, strides=strides).reshape(-1, dim, dim)
    return data_img, data_seg


class ILLUSTRIS(torch.utils.data.Dataset):
    def __init__(
            self,
            folder="illustris",
            dim=128,
            is_training=True,
    ):
        super().__init__()
        self.imgs, self.segs = load_illustris(folder=folder, dim=dim)
        self.illustris_noisy = load_illustris(folder=folder, dim=dim)

        self.is_training = is_training
        self.set = np.arange(0, len(self.imgs))
        # np.random.shuffle(self.set)
        self.trainset = self.set[:int(len(self.set) * 0.8)]
        self.validset = self.set[int(len(self.set) * 0.8):]
        self.topo = False

    def __len__(self):
        if self.is_training:
            return len(self.trainset)
        else:
            return len(self.validset)

    def __getitem__(self, index):
        if self.is_training:
            index = self.trainset[index]
        else:
            index = self.validset[index]
        img = self.imgs[index]
        seg = self.segs[index]

        if self.topo:
            topo = persistent_image_core(img)
            topo = topo / np.linalg.norm(topo)
            img = np.stack([topo, img])
        else:
            img = np.expand_dims(img, axis=0)
        seg = np.expand_dims(seg, axis=0)

        return img, seg

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = ILLUSTRIS(folder="../../datasets")
    print(len(dataset))
    img, seg = dataset[0]
    print(img.min(), img.max())
    print(seg.min(), seg.max())

    plt.imshow(img[0])
    plt.show()
    plt.imshow(seg[0])
    plt.show()
