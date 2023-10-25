from typing import Union

from torch.utils.data import Dataset
import torch

from feature_extractor import ViTFeatureExtractor
from pathlib import WindowsPath

from main.seg_classification.cnns.cnn_utils import convnet_resize_transform, convnet_preprocess
from utils import get_image_from_path
from utils.transformation import resize
from utils.vit_utils import get_image_and_inputs_and_transformed_image


class ImageSegOptDataset(Dataset):
    def __init__(
            self,
            image_path: Union[str, WindowsPath],
            target: int,
            is_explaniee_convnet: bool,
            is_competitive_method_transforms: bool,
            feature_extractor: ViTFeatureExtractor = None,
    ):
        self.image_path = image_path
        self.target = target
        self.feature_extractor = feature_extractor
        self.is_explaniee_convnet = is_explaniee_convnet
        self.is_competitive_method_transforms = is_competitive_method_transforms

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        image = get_image_from_path(path=self.image_path)
        image = image if image.mode == "RGB" else image.convert("RGB")  # Black & White images
        if not self.is_explaniee_convnet:
            inputs, resized_and_normalized_image = get_image_and_inputs_and_transformed_image(
                image=image, feature_extractor=self.feature_extractor,
                is_competitive_method_transforms=self.is_competitive_method_transforms
            )
            image_resized = resize(image)
            inputs = inputs["pixel_values"]
        else:
            inputs = convnet_preprocess(image)
            resized_and_normalized_image = convnet_preprocess(image)
            image_resized = convnet_resize_transform(image)

        return dict(
            image_name=self.image_path.split('/')[-1].split('.')[0],
            pixel_values=inputs,
            resized_and_normalized_image=resized_and_normalized_image,
            image=image_resized,
            target_class=torch.tensor(self.target),
        )
