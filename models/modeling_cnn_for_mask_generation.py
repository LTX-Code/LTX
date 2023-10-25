from icecream import ic
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchvision.models import DenseNet, ResNet


class CNNForMaskGeneration(LightningModule):
    def __init__(self, cnn_model, activation_function: str = "sigmoid", img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        self.activation_function = activation_function
        backbone_children = list(cnn_model.children())
        if type(cnn_model) is DenseNet:
            self.encoder = nn.Sequential(list(backbone_children)[0])  # output shape: [batch_size, 1920, 7, 7]
            input_features = list(backbone_children)[1].in_features
        elif type(cnn_model) is ResNet:
            self.encoder = nn.Sequential(*backbone_children[:-2])  # output shape: [batch_size, 2048, 7, 7]
            input_features = backbone_children[-1].in_features
        else:
            raise (NotImplementedError)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=input_features, out_channels=1, kernel_size=1, stride=1),
        )

    def forward(self, inputs):  # inputs.shape: [batch_size, 3, 224, 224]
        batch_size = inputs.shape[0]
        self.encoder.eval()
        enc_rep = self.encoder(inputs)
        mask = self.bottleneck(enc_rep)
        if self.activation_function == 'sigmoid':
            tokens_mask = torch.sigmoid(mask)

        interpolated_mask = torch.nn.functional.interpolate(tokens_mask,
                                                            scale_factor=int(inputs.shape[-1] / mask.shape[-1]),
                                                            mode="bilinear")
        interpolated_mask = interpolated_mask.view(batch_size, 1, self.img_size, self.img_size)
        return interpolated_mask, tokens_mask  # [batch_size, 1, img_size, img_size] , [batch_size, 1, n_tokens ]
