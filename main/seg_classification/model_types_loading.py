from copy import deepcopy
from typing import Union
from torchvision.models.resnet import ResNet
from torchvision.models.densenet import DenseNet
from torchvision import models
from transformers import ViTForImageClassification
from feature_extractor import ViTFeatureExtractor
from models.modeling_cnn_for_mask_generation import CNNForMaskGeneration
from models.modeling_vit_patch_classification import ViTForMaskGeneration
from vit_loader.load_vit import load_vit_pretrained_for_explanier, load_vit_pretrained_for_explaniee

CONVNET_MODELS_BY_NAME = {"resnet": models.resnet101(pretrained=True),
                          "densenet": models.densenet201(pretrained=True),
                          }


def load_vit_type_models(model_name: str, is_explanier_model: bool) -> Union[
    ViTForImageClassification, ViTForMaskGeneration]:
    if is_explanier_model:
        if model_name in ["google/vit-base-patch16-224"]:
            model_for_classification_image = load_vit_pretrained_for_explanier(model_name=model_name)
        else:
            model_for_classification_image = ViTForImageClassification.from_pretrained(model_name)
        return model_for_classification_image
    else:
        if model_name in ["google/vit-base-patch16-224"]:
            model_for_mask_generation = load_vit_pretrained_for_explaniee(model_name=model_name)
        else:
            model_for_mask_generation = ViTForMaskGeneration.from_pretrained(model_name)
        return model_for_mask_generation


def load_convnet_type_models(model_name: str,
                             is_explanier_model: bool,
                             activation_function: str,
                             img_size: int) -> Union[
    ResNet, DenseNet, CNNForMaskGeneration]:
    model_for_classification_image = deepcopy(CONVNET_MODELS_BY_NAME[model_name].eval())
    if not is_explanier_model:
        return model_for_classification_image
    model_for_mask_generation = CNNForMaskGeneration(cnn_model=model_for_classification_image,
                                                     activation_function=activation_function, img_size=img_size)
    del model_for_classification_image
    return model_for_mask_generation


def load_model_by_name(model_name: str, is_explanier_model: bool, activation_function: str, img_size: int):
    if model_name in CONVNET_MODELS_BY_NAME.keys():
        model = load_convnet_type_models(model_name=model_name,
                                         is_explanier_model=is_explanier_model,
                                         activation_function=activation_function,
                                         img_size=img_size,
                                         )
    else:
        model = load_vit_type_models(model_name=model_name, is_explanier_model=is_explanier_model)
    return model


def load_feature_extractor(explainee_model_name: str, explainer_model_name: str) -> Union[ViTFeatureExtractor, None]:
    """
    If both of models are convnet, return None as feature extractor, else return feature extractor by the explanier / explaniee
    """
    if explainee_model_name in CONVNET_MODELS_BY_NAME.keys() and explainer_model_name in CONVNET_MODELS_BY_NAME.keys():
        return None
    if explainee_model_name not in CONVNET_MODELS_BY_NAME.keys():
        return ViTFeatureExtractor.from_pretrained(explainee_model_name)
    if explainer_model_name not in CONVNET_MODELS_BY_NAME.keys():
        return ViTFeatureExtractor.from_pretrained(explainer_model_name)


def load_explainer_explaniee_models_and_feature_extractor(explainee_model_name: str,
                                                          explainer_model_name: str,
                                                          img_size: int,
                                                          activation_function: str = 'sigmoid',
                                                          ):
    model_for_classification_image = load_model_by_name(model_name=explainee_model_name,
                                                        is_explanier_model=False,
                                                        activation_function=activation_function,
                                                        img_size=img_size,
                                                        )
    model_for_mask_generation = load_model_by_name(model_name=explainer_model_name,
                                                   is_explanier_model=True,
                                                   activation_function=activation_function,
                                                   img_size=img_size,
                                                   )
    feature_extractor = load_feature_extractor(explainee_model_name=explainee_model_name,
                                               explainer_model_name=explainer_model_name)
    return model_for_classification_image, model_for_mask_generation, feature_extractor
