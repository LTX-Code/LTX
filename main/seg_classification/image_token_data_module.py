import pytorch_lightning as pl
from torch.utils.data import DataLoader

from feature_extractor import ViTFeatureExtractor
from main.seg_classification.image_token_dataset import ImageSegDataset, ImagesDataset


class ImageSegDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            train_images_path: str,
            val_images_path: str,
            is_sampled_train_data_uniformly: bool,
            is_sampled_val_data_uniformly: bool,
            is_explaniee_convnet: bool,
            is_competitive_method_transforms: bool,
            train_n_label_sample: int,
            val_n_label_sample: int,
            feature_extractor: ViTFeatureExtractor = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_images_path = train_images_path
        self.val_images_path = val_images_path
        self.is_sampled_train_data_uniformly = is_sampled_train_data_uniformly
        self.is_sampled_val_data_uniformly = is_sampled_val_data_uniformly
        self.is_explaniee_convnet = is_explaniee_convnet
        self.is_competitive_method_transforms = is_competitive_method_transforms
        self.train_n_label_sample = train_n_label_sample
        self.val_n_label_sample = val_n_label_sample
        self.feature_extractor = feature_extractor

    def setup(self, stage=None):
        dataset = ImageSegDataset(
            images_path=self.train_images_path,
            feature_extractor=self.feature_extractor,
            is_sampled_train_data_uniformly=self.is_sampled_train_data_uniformly,
            is_sampled_val_data_uniformly=self.is_sampled_val_data_uniformly,
            train_n_label_sample=self.train_n_label_sample,
            val_n_label_sample=self.val_n_label_sample,
        )
        self.train_dataset = ImagesDataset(images_path=self.train_images_path,
                                           images_name=dataset.train_set,
                                           targets=dataset.train_gt_classes,
                                           is_explaniee_convnet=self.is_explaniee_convnet,
                                           is_competitive_method_transforms=self.is_competitive_method_transforms,
                                           feature_extractor=self.feature_extractor,
                                           )

        self.val_dataset = ImagesDataset(images_path=self.val_images_path,
                                         images_name=dataset.val_set,
                                         targets=dataset.val_gt_classes,
                                         is_explaniee_convnet=self.is_explaniee_convnet,
                                         is_competitive_method_transforms=self.is_competitive_method_transforms,
                                         feature_extractor=self.feature_extractor,
                                         )

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False),
