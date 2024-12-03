import torch
from torchvision import transforms
import numpy as np
import pixhomology as px
from PIL import Image
import random
import torchvision.transforms.functional as F


def rgb2gray(rgb):
    r, g, b = rgb[0, :,:], rgb[1, :,:], rgb[2, :,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = np.expand_dims(gray, axis=0)
    return gray


def persistent_image_core(x, maxdim=1, dgms=None, idxs=None, n_components=None, b_h=None, c_w=None):
    h, w = x.shape
    if dgms is None:
        dgms, idxs = px.computePH(x, maxdim=maxdim, return_index=True)

    img = np.zeros((h, w), dtype=np.float32)
    for dim in range(1 + maxdim):
        dgm = dgms[dim]
        idx = idxs[dim]
        persistence = dgm[:, 0] - dgm[:, 1]
        # if n_components is None:
        #    n_components = len(persistence)

        for i in range(len(idx)):  # np.flip(np.argsort(persistence))[:n_components]:
            if dim == 1:
                yd = idx[i, 0]
                xd = idx[i, 1]
                yb = idx[i, 2]
                xb = idx[i, 3]
            else:
                yb = idx[i, 0]
                xb = idx[i, 1]
                yd = idx[i, 2]
                xd = idx[i, 3]
                
            if b_h is not None and c_w is not None:
                # Determine the window containing (xb, yb)
                window_x_start = (xb // b_h) * b_h
                window_y_start = (yb // c_w) * c_w
                window_x_end = min(window_x_start + b_h, h)
                window_y_end = min(window_y_start + c_w, w)

                # Rescale dist using window limits
                if window_x_start <= xd < window_x_end:
                    xd = xd
                else:
                    xd = window_x_end - 1

                if window_y_start <= yd < window_y_end:
                    yd = yd
                else:
                    yd = window_y_end - 1

                dist = np.sqrt((xb - xd) ** 2 + (yb - yd) ** 2)

                # Constrain x_start, y_start, x_end, y_end to the window
                x_start = int(max(window_x_start, xb - 2 * dist))
                y_start = int(max(window_y_start, yb - 2 * dist))
                x_end = int(min(window_x_end, xb + 2 * dist))
                y_end = int(min(window_y_end, yb + 2 * dist))
            else:
                dist = np.sqrt((xb - xd) ** 2 + (yb - yd) ** 2)

                x_start = int(max(0, min(h - 1, xb - 2 * dist)))
                y_start = int(max(0, min(w - 1, yb - 2 * dist)))
                x_end = int(max(0, min(h, xb + 2 * dist)))
                y_end = int(max(0, min(w, yb + 2 * dist)))

            # Generate grid of coordinates for the entire image
            dx = np.arange(x_start, x_end)[:, None] - xb
            dy = np.arange(y_start, y_end)[None, :] - yb

            img[x_start:x_end, y_start:y_end] += ((-1) ** dim) * persistence[i] * np.exp(
                -(dx ** 2 + dy ** 2) / (np.sqrt(dist) + 1e-05))
    return img


def persistent_image(x, maxdim=1):
    out = []
    for img in x:
        img = persistent_image_core(img, maxdim=maxdim)
        out.append(img)
    return np.stack(out)



class TopoTransform:
    def __init__(self, ):
        pass
    def __call__(self, image):
        # Convert to tensor and normalize (you can customize this part as needed)
        if image.shape[0] == 3:
            topo = rgb2gray(image)
        else:
            topo = image
        if topo.min() == topo.max():
            topo = np.ones_like(topo)
        else:
            topo = (topo - topo.min()) / (topo.max() - topo.min())
        topo = persistent_image(topo)
        x = np.concatenate([topo, image], axis=0)
        x = x.astype(np.float32)

        return x
