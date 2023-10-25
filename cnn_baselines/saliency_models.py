import captum
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from captum.attr import DeepLift
from torchvision.models import ConvNeXt_Base_Weights
import numpy as np
from cnn_baselines.grad_methods_utils import blend_image_and_heatmap, get_torchgc_model_layer
from cnn_baselines.torchgc.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from main.seg_classification.cnns.cnn_utils import CONVENT_NORMALIZATION_MEAN, CONVNET_NORMALIZATION_STD


class GradModel(nn.Module):
    def __init__(self, model_name: str, feature_layer: int):
        super(GradModel, self).__init__()
        self.post_features = None
        self.model_str = 'None'
        if model_name == 'resnet101':
            model = create_resnet101_module(requires_grad=True, pretrained=True)
            self.features = model[:feature_layer]
            self.post_features = model[feature_layer:-1]
            self.avgpool = model[feature_layer:]
            self.classifier = torchvision.models.resnet101(pretrained=True).fc
        elif model_name == 'densenet':
            model = torchvision.models.densenet201(pretrained=True)
            model.eval()
            self.features = model.features[:feature_layer]
            self.post_features = model.features[feature_layer:-1]
            self.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=1)
            self.classifier = model.classifier
        elif model_name == 'convnext':
            self.model_str = model_name
            model = torchvision.models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
            self.features = model.features
            self.avgpool = model.avgpool
            self.classifier = model.classifier
        else:
            raise NotImplementedError

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad.detach().clone()

    def activations_hook_t(self, grad):
        self.gradients_t = grad.detach().clone()

    def activations_hook_s(self, grad):
        self.gradients_s = grad.detach().clone()

    def get_activations_gradient(self):
        return self.gradients

    def get_activations_gradient_t(self):
        return self.gradients_t

    def get_activations_gradient_s(self):
        return self.gradients_s

    def get_activations(self, x):
        x = self.features(x)
        return x

    def get_post_activations(self, x):
        x = self.post_features(x)
        return x

    def compute_representation(self, x, hook=True):
        x = self.forward_(x, hook)
        return x

    def compute_t_s_representation(self, t, s, only_post_features=False):
        t = self.forward_(t, True, only_post_features, hook_func=lambda grad: self.activations_hook_t(grad))
        s = self.forward_(s, True, only_post_features, hook_func=lambda grad: self.activations_hook_s(grad))

        return t, s

    def activations_to_features(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_(self, x, hook=True, only_post_features=False, hook_func=None):
        if not only_post_features:
            x = self.features(x)

        if hook:
            hook_func = self.activations_hook if hook_func is None else hook_func
            x.register_hook(hook_func)

        if self.post_features is not None:
            x = self.post_features(x)

        if self.model_str == 'convnext':
            x = F.relu(x)
            x = self.avgpool(x)
        else:
            x = F.relu(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        return x

    def forward(self, x, hook=True, only_post_features=False):
        x = self.forward_(x, hook=hook, only_post_features=only_post_features)
        x = self.classifier(x)
        return x


class ReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input, inplace=False)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def create_sequential_model_without_top(model, num_top_layers=1):
    model_children_without_top_layers = list(model.children())[:-num_top_layers]
    return nn.Sequential(*model_children_without_top_layers)


def create_resnet101_module(pretrained=True, requires_grad=False):
    model = torchvision.models.resnet101(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = requires_grad

    return create_sequential_model_without_top(model)


def lift_cam(model,
             inputs,
             label,
             device,
             use_mask: bool,
             ):
    model.eval()
    model.zero_grad()
    output = model(inputs.to(device), hook=True)

    class_id = label
    if class_id is None:
        class_id = torch.argmax(output, dim=1)

    # act_map = model.get_activations(inputs.to(device)).detach().cpu()
    act_map = model.get_activations(inputs.to(device))

    model_part = Model_Part(model)
    model_part.eval()
    dl = DeepLift(model_part)
    ref_map = torch.zeros_like(act_map).to(device)
    dl_contributions = dl.attribute(act_map, ref_map, target=class_id, return_convergence_delta=False).detach()

    scores_temp = torch.sum(dl_contributions, (2, 3), keepdim=False).detach()
    scores = torch.squeeze(scores_temp, 0)
    scores = scores.cpu()

    vis_ex_map = (scores[None, :, None, None] * act_map.cpu()).sum(dim=1, keepdim=True)
    vis_ex_map = F.relu(vis_ex_map).float()

    with torch.no_grad():
        heatmap = vis_ex_map
        heatmap = F.interpolate(heatmap, scale_factor=int(224 / 7), mode="bilinear")
        # heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        heatmap = heatmap.squeeze().cpu().data.numpy()
        # t = tensor2cv(inputs.squeeze(0))
        # blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t,
        #                                                                                    heatmap,
        #                                                                                    use_mask=use_mask)
    return heatmap
    # return t, blended_img, heatmap_cv, blended_img_mask, inputs.squeeze(0), score, heatmap


def tensor2cv(inp):
    inp = inp.cpu().detach().numpy().transpose((1, 2, 0))
    mean = np.array(CONVENT_NORMALIZATION_MEAN)
    std = np.array(CONVNET_NORMALIZATION_STD)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = np.uint8(255 * inp)
    return inp


def generic_torchcam(modelCAM,
                     backbone_name: str,
                     inputs,
                     label,
                     device,
                     use_mask: bool
                     ):
    model, layer = get_torchgc_model_layer(backbone_name, device)
    cam_extractor = modelCAM(model, layer)
    targets = [ClassifierOutputTarget(label)]
    hm = cam_extractor(inputs.to(device), targets)

    heatmap = torch.tensor(hm).unsqueeze(0)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().detach().cpu().data.numpy()
    return heatmap
    # t = tensor2cv(inputs.squeeze(0))
    # blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(img_cv=t,
    #                                                                                    heatmap=heatmap,
    #                                                                                    use_mask=use_mask,
    #                                                                                    )

    # return t, blended_img, heatmap_cv, blended_img_mask, inputs.squeeze(0), score, heatmap


def ig_captum(model,
              inputs,
              label,
              device,
              use_mask: bool,
              ):
    model.eval()
    model.zero_grad()
    class_id = label

    integrated_grads = captum.attr.IntegratedGradients(model)
    baseline = torch.zeros_like(inputs).to(device)
    attr = integrated_grads.attribute(inputs.to(device), baseline, class_id)

    with torch.no_grad():
        heatmap = torch.mean(attr, dim=1, keepdim=True) # IG already generate map [224,224]
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        heatmap = heatmap.squeeze().cpu().data.numpy()
        t = tensor2cv(inputs.squeeze(0))
        blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t,
                                                                                           heatmap,
                                                                                           use_mask=use_mask,
                                                                                            )
    return t, blended_img, heatmap_cv, blended_img_mask, inputs.squeeze(0), score, heatmap


class Model_Part(nn.Module):
    def __init__(self, model):
        super(Model_Part, self).__init__()
        self.model_type = None
        if model.model_str == 'convnext':
            self.avg_pool = model.avgpool
            self.classifier = model.classifier[-1]
        else:
            self.avg_pool = model.avgpool
            self.classifier = model.classifier

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
