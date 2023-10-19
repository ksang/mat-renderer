import os
import pathlib
import math

import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
import PIL

import torchvision.transforms.functional as F


from matrenderer import IMAGE_EXTENSIONS, SVBRDF_MAPS


def read_image(filename: str) -> torch.Tensor:
    """Read a local image file into a float tensor (pixel values are normalized to [0, 1], CxHxW)

    Args:
        filename: Image file path.

    Returns:
        Loaded image tensor.
    """
    img: np.ndarray = imageio.imread(filename)

    # Convert the image array to float tensor according to its data type
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16 or img.dtype == np.int32:
        img = img.astype(np.float32) / 65535.0
    else:
        raise ValueError(f'Unrecognized image pixel value type: {img.dtype}')

    if img.ndim == 2:
        return torch.from_numpy(img).unsqueeze(0)  # 1xHxW for grayscale images
    elif img.ndim == 3:
        return torch.from_numpy(img).movedim(2, 0) # HxWxC to CxHxW
    else:
        raise ValueError(f'Unrecognized image dimension: {img.shape}')
    
def get_image_filenames(path: str) -> [str]:
    """Get SVBRDF image filenames from a given path, supported image extension and SVBRDF map names are defined in
        IMAGE_EXTENSIONS and SVBRDF_MAPS.

    Args:
        path: Material SVBRDF images path.

    Returns:
        Dictionary that describes map names and filenames.
    """
    res = dict.fromkeys(SVBRDF_MAPS)
    for file in os.listdir(path):
        if pathlib.Path(file).suffix.lower() in IMAGE_EXTENSIONS:
            for map_name in SVBRDF_MAPS:
                if pathlib.Path(file).stem.lower().endswith(map_name):
                    if res[map_name]:
                        print("multiple svBRDF maps found, replacing: {} -> {}".format(res[map_name], file))
                    res[map_name] = os.path.join(path, file)
                    break
    return res

def load_svbrdf_maps(path: str) -> {str: torch.Tensor}:
    """Get SVBRDF maps from a given path, supported image extension and SVBRDF map names are defined in
        IMAGE_EXTENSIONS and SVBRDF_MAPS.

    Args:
        path: Material SVBRDF images path.

    Returns:
        Dictionary that describes map name and data.
    """
    res = dict.fromkeys(SVBRDF_MAPS)
    filenames = get_image_filenames(path)
    for map_name in filenames.keys():
        if filenames[map_name]:
            res[map_name] = read_image(filenames[map_name])

    return res

def show_maps(maps: {str: torch.tensor}, fig_width: int=10, ncols: int=2):
    n_maps = 0
    for m in maps.values():
        if m is not None: n_maps += 1

    nrows = math.ceil(n_maps / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, int(fig_width/2)*nrows), squeeze=False)

    for i, map_name in enumerate(maps.keys()):
        if maps[map_name] is None:
            continue

        tensor_img = maps[map_name].detach()
        img = F.to_pil_image(tensor_img)

        if tensor_img.shape[0] == 1:
            axs[int(i/ncols), i%ncols].imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            axs[int(i/ncols), i%ncols].imshow(img)
        axs[int(i/ncols), i%ncols].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[int(i/ncols), i%ncols].set_title(map_name)
    
    if len(maps.keys()) < ncols * nrows:
        axs[nrows-1, ncols-1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    fig.tight_layout()

def show_rendered(image: torch.tensor, figsize=(8, 8), title=None):
    fig = plt.figure(figsize=figsize)
    if title:
        fig.suptitle(title)
    plt.axis('off')
    plt.imshow((image.permute(1,2,0)).cpu().detach().numpy())