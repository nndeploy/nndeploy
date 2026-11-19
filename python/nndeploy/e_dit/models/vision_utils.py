"""Image-to-tensor conversion shared by conditioning nodes."""
import numpy as np
import torch
from PIL import Image

def load_image(path):
    if isinstance(path, Image.Image):
        return path.convert('RGB')
    with Image.open(path) as image:
        return image.convert('RGB')
def pil_to_numpy(images):
    if not isinstance(images, list): images = [images]
    return np.stack([np.asarray(im).astype(np.float32) / 255 for im in images])
def numpy_to_pt(images):
    return torch.from_numpy(images.transpose(0, 3, 1, 2))
def normalize(images):
    return 2 * images - 1
