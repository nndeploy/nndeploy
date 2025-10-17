
from typing import List, Optional, Tuple, Union
import numpy as np
# from PIL import Image
from PIL import Image
import torch

ImageInput = Union[
    Image.Image,
    np.ndarray,
    torch.Tensor,
    List[Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]