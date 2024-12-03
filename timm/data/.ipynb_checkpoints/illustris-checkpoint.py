from numpy.lib.stride_tricks import as_strided
from astropy.io import fits
import os
import numpy as np
import torch
import scipy

from ._topo import persistent_image_core

def load_illustris(folder=".", dim=128, noise=True, sigma=0.1):
    hdu = fits.open(os.path.join(folder, "illustris", "illustris.fits"))
    arr = hdu[3].data.astype(np.float32)
    if noise:
        gaussian_noise = scipy.stats.norm.rvs(loc=0, scale=sigma, size=arr.shape, random_state=42)
        arr += sigma * gaussian_noise

    p = np.percentile(arr, 99)
    arr = np.clip(arr, None, p) / p
    #arr = arr / arr.max()

    window_shape = (dim, dim)
    output_shape = (arr.shape[0] // window_shape[0], arr.shape[1] // window_shape[1]) + window_shape
    strides = (arr.strides[0] * window_shape[0], arr.strides[1] * window_shape[1]) + arr.strides
    data = as_strided(arr, shape=output_shape, strides=strides).reshape(-1, dim, dim)
    return data


class ILLUSTRIS(torch.utils.data.Dataset):
    def __init__(
            self,
            folder="illustris",
            dim=128,
            is_training=True,
            sigma=0.1,
    ):
        super().__init__()
        self.illustris_clean = load_illustris(folder=folder, dim=dim, noise=False, sigma=sigma)
        self.illustris_noisy = load_illustris(folder=folder, dim=dim, noise=True, sigma=sigma)

        self.is_training = is_training
        self.set = np.arange(0, len(self.illustris_clean))
        #np.random.shuffle(self.set)
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
        img = self.illustris_clean[index]
        noisy = self.illustris_noisy[index]

        if self.topo:
            topo = persistent_image_core(noisy)
            noisy = np.stack([topo, noisy])
        else:
            noisy = np.expand_dims(noisy, axis=0)

        img = np.expand_dims(img, axis=0)

        return noisy, img

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = ILLUSTRIS(folder=".", sigma=0.1)
    print(len(dataset))
    print(dataset[0][0].min(), dataset[0][0].max())
    print(dataset[0][1].min(), dataset[0][1].max())
    plt.imshow(dataset[0][0][0])
    plt.show()
    plt.imshow(dataset[0][1][0])
    plt.show()