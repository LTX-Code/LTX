from typing import List
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm import tqdm

from main.seg_classification.cnns.cnn_utils import CONVNET_NORMALIZATION_STD, CONVENT_NORMALIZATION_MEAN
from utils import show_image

device = torch.device('cuda')


def run_all_operations(model,
                       image_preprocessed,
                       label,
                       backbone_name: str,
                       device,
                       features_layer: int,
                       methods: List[str],
                       use_mask: bool = False,
                       ):
    results = []
    for method in methods:
        t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap = run_by_class_grad(model=model,
                                                                                              image_preprocessed=image_preprocessed,
                                                                                              label=label,
                                                                                              backbone_name=backbone_name,
                                                                                              device=device,
                                                                                              features_layer=features_layer,
                                                                                              method=method,
                                                                                              use_mask=use_mask,
                                                                                              )
        results.append((t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap))

        show_image(blended_img, title=method)
    return results


def run_by_class_grad(model,
                      image_preprocessed,
                      label,
                      backbone_name: str,
                      device,
                      features_layer: int,
                      method: str,
                      use_mask: bool = False
                      ):
    label = torch.tensor(label, dtype=torch.long, device=device)
    t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap = by_class_map(model=model,
                                                                                     image=image_preprocessed,
                                                                                     label=label,
                                                                                     method=method,
                                                                                     use_mask=use_mask)

    return heatmap
    # return t1, blended_img, heatmap_cv, blended_img_mask, image_preprocessed, score, heatmap


def by_class_map(model, image, label, method: str, use_mask=False):
    weight_ratio = []
    model.eval()
    model.zero_grad()
    preds = model(image.unsqueeze(0).to(device), hook=True)
    _, predicted = torch.max(preds.data, 1)
    # print(f'True label {label}, predicted {predicted}')

    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()
    preds.to(device)
    one_hot.to(device)
    gradients = model.get_activations_gradient()
    heatmap = grad2heatmaps(model=model,
                            X=image.unsqueeze(0).to(device),
                            gradients=gradients,
                            activations=None,
                            method=method,
                            score=score,
                            do_nrm_rsz=True,
                            weight_ratio=weight_ratio)

    t = tensor2cv(image)
    blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(img_cv=t,
                                                                                       heatmap=heatmap,
                                                                                       use_mask=use_mask,
                                                                                       )

    return t, blended_img, heatmap_cv, blended_img_mask, t, score, heatmap


def tensor2cv(inp):
    inp = inp.cpu().detach().numpy().transpose((1, 2, 0))
    mean = np.array(CONVENT_NORMALIZATION_MEAN)
    std = np.array(CONVNET_NORMALIZATION_STD)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = np.uint8(255 * inp)
    return inp


def grad2heatmaps(model,
                  X,
                  gradients,
                  activations=None,
                  method='iig'
                  , score=None,
                  do_nrm_rsz=True,
                  weight_ratio=[],
                  ):
    if activations is None:
        activations = model.get_activations(
            X).detach()  # get the features of the model (some kind of feature extractor)

    if method == 'iig':
        act_grads = F.relu(activations) * F.relu(gradients.detach()) ** 2
        heatmap = torch.sum(act_grads.squeeze(0), dim=0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif method == 'gradcam':
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
        heatmap = torch.mean(activations * pooled_gradients, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
    elif method == 'x-gradcam':
        sum_activations = np.sum(activations.detach().cpu().numpy(), axis=(2, 3))
        eps = 1e-7
        weights = gradients.detach().cpu().numpy() * activations.detach().cpu().numpy() / \
                  (sum_activations[:, :, None, None] + eps)
        weights = weights.sum(axis=(2, 3))
        weights_tensor = torch.tensor(weights).unsqueeze(2).unsqueeze(3)
        heatmap = F.relu(
            torch.sum(gradients.detach().cpu() * weights_tensor.detach().cpu() * activations.detach().cpu(), dim=1,
                      keepdim=True))
    elif method == 'activations':
        heatmap = torch.sum(F.relu(activations).squeeze(0), dim=0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif method == 'gradients':
        heatmap = torch.sum((F.relu(gradients.detach())).squeeze(0), dim=0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif method == 'neg_gradients':
        heatmap = torch.sum((F.relu(-1 * gradients.detach())).squeeze(0), dim=0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif method == 'gradcampp':
        gradients = gradients.detach()
        activations = activations.detach()
        score = score.detach()
        square_grad = gradients.pow(2)
        denominator = 2 * square_grad + activations.mul(gradients.pow(3)).sum(dim=[2, 3], keepdim=True)
        denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        alpha = torch.div(square_grad, denominator + 1e-6)
        pos_grads = F.relu(score.exp() * gradients).detach()
        weights = torch.sum(alpha * pos_grads, dim=[2, 3], keepdim=True).detach()
        heatmap = torch.sum(activations * weights, dim=1, keepdim=True).detach()

    # heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
    heatmap = F.interpolate(heatmap, scale_factor=int(224 / 7), mode="bilinear")
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().cpu().data.numpy()

    return heatmap


def blend_image_and_heatmap(img_cv, heatmap, use_mask=False):
    heatmap -= np.min(heatmap)

    if heatmap.max() != torch.tensor(0.):
        heatmap /= heatmap.max()

    blended_img_mask = None

    if use_mask:
        score = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap_cv = np.uint8(score)
        blended_img_mask = np.uint8((np.repeat(score.reshape(224, 224, 1), 3, axis=2) * img_cv))

    heatmap = np.max(heatmap) - heatmap
    if np.max(heatmap) < 255.:
        heatmap *= 255

    score = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_cv = np.uint8(score)
    heatmap_cv = cv2.applyColorMap(heatmap_cv, cv2.COLORMAP_JET)

    blended_img = heatmap_cv * 0.9 + img_cv
    blended_img = cv2.normalize(blended_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    blended_img[blended_img < 0] = 0

    return blended_img, score, heatmap_cv, blended_img_mask, img_cv


def get_torchgc_model_layer(backbone_name: str, device):
    if backbone_name.__contains__('resnet'):
        resnet101 = torchvision.models.resnet101(pretrained=True).to(device)
        resnet101_layer = resnet101.layer4
        return resnet101, resnet101_layer
    elif backbone_name.__contains__('convnext'):
        convnext = torchvision.models.convnext_base(pretrained=True).to(device)
        convnext_layer = convnext.features[-1]
        return convnext, convnext_layer
    elif backbone_name.__contains__('densenet'):
        densnet201 = torchvision.models.densenet201(pretrained=True).to(device)
        densnet201_layer = densnet201.features
        return densnet201, densnet201_layer
    else:
        raise NotImplementedError
