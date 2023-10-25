import pickle
from pathlib import Path, WindowsPath
from typing import Tuple, Union, Any, List
import torch
from torch import Tensor
from sklearn.metrics import auc
import os
import numpy as np
from config import config


def load_obj_from_path(path: Union[str, WindowsPath, Path]) -> Any:
    if type(path) == str and path[-4:] != '.pkl':
        path += '.pkl'
    elif type(path) == WindowsPath and path.suffix != '.pkl':
        path = path.with_suffix('.pkl')

    with open(path, 'rb') as f:
        return pickle.load(f)


def patch_score_to_image(transformer_attribution: Tensor,
                         output_2d_tensor: bool = True,
                         img_size: int = 224,
                         patch_size: int = 16,
                         ) -> Tensor:
    """
    Convert Patch scores ([196]) to image size tesnor [224, 224]
    :param transformer_attribution: Tensor with score of each patch in the picture
    :return:
    """
    transformer_attribution = transformer_attribution.reshape(1, 1,
                                                              int(img_size / patch_size),
                                                              int(img_size / patch_size))
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    if output_2d_tensor:
        transformer_attribution = transformer_attribution.reshape(img_size, img_size).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())
    return transformer_attribution


def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def calculate_auc(mean_accuracy_by_step: np.ndarray) -> float:
    return auc(x=np.arange(0, 1, 0.1), y=mean_accuracy_by_step)


def _remove_file_if_exists(path: Path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
