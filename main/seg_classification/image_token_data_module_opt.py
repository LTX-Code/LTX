import pytorch_lightning as pl
from torch.utils.data import DataLoader
from feature_extractor import ViTFeatureExtractor
from main.seg_classification.image_token_opt_dataset import ImageSegOptDataset


class ImageSegOptDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            train_image_path: str,
            val_image_path: str,
            target: int,
            is_explaniee_convnet: bool,
            is_competitive_method_transforms: bool = False,
            feature_extractor: ViTFeatureExtractor = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_image_path = train_image_path
        self.val_image_path = val_image_path
        self.target = target
        self.is_explaniee_convnet = is_explaniee_convnet
        self.is_competitive_method_transforms = is_competitive_method_transforms
        self.feature_extractor = feature_extractor

    def setup(self, stage=None):
        self.val_dataset = ImageSegOptDataset(
            image_path=self.val_image_path,
            target=self.target,
            is_explaniee_convnet=self.is_explaniee_convnet,
            is_competitive_method_transforms=self.is_competitive_method_transforms,
            feature_extractor=self.feature_extractor,
        )
        self.train_dataset = ImageSegOptDataset(
            image_path=self.train_image_path,
            target=self.target,
            is_explaniee_convnet=self.is_explaniee_convnet,
            is_competitive_method_transforms=self.is_competitive_method_transforms,
            feature_extractor=self.feature_extractor,
        )

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False)
