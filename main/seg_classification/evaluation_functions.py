import os
from icecream import ic
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
import pickle
from typing import Union, Dict, List, Tuple
from torchvision.transforms import transforms
from tqdm import tqdm
from transformers import ViTForImageClassification
from config import config
from pytorch_lightning import seed_everything
from torch.nn import functional as F
import numpy as np
from evaluation.perturbation_tests.seg_cls_perturbation_tests import eval_perturbation_test
from main.seg_classification.backbone_to_details import EXPLAINER_EXPLAINEE_BACKBONE_DETAILS
from main.seg_classification.cnns.cnn_utils import CONVENT_NORMALIZATION_MEAN, CONVNET_NORMALIZATION_STD, \
    convnet_resize_transform
from main.seg_classification.model_types_loading import load_explainer_explaniee_models_and_feature_extractor, \
    CONVNET_MODELS_BY_NAME
from utils.consts import IMAGENET_VAL_IMAGES_FOLDER_PATH, GT_VALIDATION_PATH_LABELS, MODEL_ALIAS_MAPPING
import torch
from enum import Enum
import pickle

vit_config = config["vit"]

seed_everything(config['general']['seed'])
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def get_gt_classes(path):
    with open(path, 'r') as f:
        gt_classes_list = f.readlines()
    gt_classes_list = [int(record.split()[-1].replace('\n', '')) for record in gt_classes_list]
    return gt_classes_list


class PerturbationType(Enum):
    POS = "POS"
    NEG = "NEG"


def load_obj(path):
    with open(Path(path), 'rb') as f:
        return pickle.load(f)


def get_image(path) -> Image:
    image = Image.open(path)
    return image


resize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def plot_image(image, title=None) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    if title is not None:
        plt.title(title)
    plt.axis('off');
    plt.show();


def show_mask(mask, model_type='N/A', auc='N/A'):  # [1, 1, 224, 224]
    mask = mask if len(mask.shape) == 3 else mask.squeeze(0)
    _ = plt.imshow(mask.squeeze(0).cpu().detach())
    if model_type != "N/A" or auc != "N/A":
        plt.title(f'{model_type}, auc: {auc}')
    plt.axis('off');
    plt.show()
    return


def get_normalization_mean_std(is_explainee_convnet: bool) -> Tuple[List[float], List[float]]:
    mean, std = (CONVENT_NORMALIZATION_MEAN, CONVNET_NORMALIZATION_STD) if is_explainee_convnet else (
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5])
    return mean, std


def normalize_mask_values(mask, clamp_between_0_to_1: bool = False):
    if clamp_between_0_to_1:
        norm_mask = torch.clamp(mask, min=0, max=1)
    else:
        norm_mask = (mask - mask.min()) / (mask.max() - mask.min())
    return norm_mask


def scatter_image_by_mask(image, mask):
    return image * mask


def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def get_probability_by_logits(logits):
    return F.softmax(logits, dim=1)[0]


def calculate_average_change_percentage(full_image_confidence: float,
                                        saliency_map_confidence: float,
                                        ) -> float:
    """
    Higher is better
    """
    return (saliency_map_confidence - full_image_confidence) / full_image_confidence


def calculate_avg_drop_percentage(full_image_confidence: float,
                                  saliency_map_confidence: float,
                                  ) -> float:
    """
    Lower is better
    """
    return max(0, full_image_confidence - saliency_map_confidence) / full_image_confidence


def calculate_percentage_increase_in_confidence(full_image_confidence: float,
                                                saliency_map_confidence: float,
                                                ) -> float:
    """
    Higher is better
    """
    return 1 if full_image_confidence < saliency_map_confidence else 0


def read_image_and_mask_from_pickls_by_path(image_path,
                                            mask_path,
                                            device,
                                            is_explainee_convnet: bool,
                                            ):
    masks_listdir = os.listdir(mask_path)
    for idx in range(len(masks_listdir)):
        pkl_path = Path(mask_path, f"{idx}.pkl")  # pkl are zero-based
        loaded_obj = load_obj(pkl_path)
        image = get_image(Path(image_path, f'ILSVRC2012_val_{str(idx + 1).zfill(8)}.JPEG'))  # images are one-based
        image = image if image.mode == "RGB" else image.convert("RGB")
        image_resized = resize(image)
        yield dict(image_resized=image_resized.unsqueeze(0).to(device),
                   image_mask=loaded_obj["vis"].to(device),
                   auc=loaded_obj["auc"],
                   )


def infer_perturbation_tests(images_and_masks,
                             model_for_image_classification,
                             perturbation_config: Dict[str, Union[PerturbationType, bool]],
                             gt_classes_list: List[int],
                             is_explainee_convnet: bool,
                             ) -> Tuple[List[float], List[float]]:
    """
    :param config: contains the configuration of the perturbation test:
        * neg: True / False
    """
    aucs_perturbation = []
    aucs_auc_deletion_insertion = []
    perturbation_type = perturbation_config["perturbation_type"].name
    is_calculate_deletion_insertion = perturbation_config["is_calculate_deletion_insertion"]
    for image_idx, image_and_mask in tqdm(enumerate(images_and_masks)):
        image, mask = image_and_mask["image_resized"], image_and_mask["image_mask"]  # [1,3,224,224], [1,1,224,224]
        outputs = [
            {'image_resized': image,
             'image_mask': mask,
             'target_class': torch.tensor([gt_classes_list[image_idx]]),
             }
        ]
        auc_perturbation, auc_deletion_insertion = eval_perturbation_test(experiment_dir=Path(""),
                                                                          model=model_for_image_classification,
                                                                          outputs=outputs,
                                                                          perturbation_type=perturbation_type,
                                                                          is_calculate_deletion_insertion=is_calculate_deletion_insertion,
                                                                          is_convenet=is_explainee_convnet,
                                                                          )
        aucs_perturbation.append(auc_perturbation)
        aucs_auc_deletion_insertion.append(auc_deletion_insertion)
    return aucs_perturbation, aucs_auc_deletion_insertion


def get_probability_and_class_idx_by_index(logits, index: int) -> float:
    probability_distribution = F.softmax(logits[0], dim=-1)
    predicted_probability_by_idx = probability_distribution[index].item()
    return predicted_probability_by_idx


def run_evaluation_metrics(model_for_image_classification,
                           inputs,
                           inputs_scatter,
                           gt_class: int,
                           is_explainee_convnet: bool
                           ):
    full_image_probability_by_index = get_probability_and_class_idx_by_index(
        logits=model_for_image_classification(inputs) if is_explainee_convnet else model_for_image_classification(
            inputs).logits,
        index=gt_class)
    saliency_map_probability_by_index = get_probability_and_class_idx_by_index(
        logits=model_for_image_classification(
            inputs_scatter) if is_explainee_convnet else model_for_image_classification(
            inputs_scatter).logits,
        index=gt_class)

    avg_drop_percentage = calculate_avg_drop_percentage(
        full_image_confidence=full_image_probability_by_index,
        saliency_map_confidence=saliency_map_probability_by_index)

    percentage_increase_in_confidence_indicators = calculate_percentage_increase_in_confidence(
        full_image_confidence=full_image_probability_by_index,
        saliency_map_confidence=saliency_map_probability_by_index
    )

    return dict(avg_drop_percentage=avg_drop_percentage,
                percentage_increase_in_confidence_indicators=percentage_increase_in_confidence_indicators,
                )


def infer_adp_pic(model_for_image_classification: ViTForImageClassification,
                  images_and_masks,
                  gt_classes_list: List[int],
                  is_explainee_convnet: bool,
                  ):
    adp_values, pic_values = [], []

    for image_idx, image_and_mask in tqdm(enumerate(images_and_masks), total=len(gt_classes_list)):
        image, mask = image_and_mask["image_resized"], image_and_mask["image_mask"]  # [1,3,224,224], [1,1,224,224]
        normalize_mean, normalize_std = get_normalization_mean_std(is_explainee_convnet=is_explainee_convnet)

        norm_original_image = normalize(image.clone(), mean=normalize_mean, std=normalize_std)
        scattered_image = scatter_image_by_mask(image=image, mask=mask)
        norm_scattered_image = normalize(scattered_image.clone(), mean=normalize_mean, std=normalize_std)
        metrics = run_evaluation_metrics(model_for_image_classification=model_for_image_classification,
                                         inputs=norm_original_image,
                                         inputs_scatter=norm_scattered_image,
                                         gt_class=gt_classes_list[image_idx],
                                         is_explainee_convnet=is_explainee_convnet,
                                         )
        adp_values.append(metrics["avg_drop_percentage"])
        pic_values.append(metrics["percentage_increase_in_confidence_indicators"])

    averaged_drop_percentage = 100 * np.mean(adp_values)
    percentage_increase_in_confidence = 100 * np.mean(pic_values)

    return dict(percentage_increase_in_confidence=percentage_increase_in_confidence,
                averaged_drop_percentage=averaged_drop_percentage,
                )


def run_evaluations(pkl_path,
                    is_base_model: bool,
                    target_or_predicted_model: str,
                    backbone_name: str,
                    imagenet_val_images_folder_path,
                    device,
                    is_explainee_convnet: bool,
                    opt_metric_type: str,
                    ):
    print(f"backbone_name: {backbone_name}")
    print(f"is_base_model: {is_base_model}")
    print(f"pkl_path: {pkl_path}")
    print(f"opt_metric_type: {opt_metric_type}")

    NAME = f'{"Base" if is_base_model else "Opt"} Model + {target_or_predicted_model} - {backbone_name}'
    print(NAME)
    images_and_masks = read_image_and_mask_from_pickls_by_path(image_path=imagenet_val_images_folder_path,
                                                               mask_path=pkl_path,
                                                               device=device,
                                                               is_explainee_convnet=is_explainee_convnet,
                                                               )

    # ADP & PIC metrics
    evaluation_metrics = infer_adp_pic(model_for_image_classification=model_for_classification_image,
                                       images_and_masks=images_and_masks,
                                       gt_classes_list=gt_classes_list,
                                       is_explainee_convnet=is_explainee_convnet,
                                       )
    print(
        f'PIC (% Increase in Confidence - Higher is better): {round(evaluation_metrics["percentage_increase_in_confidence"], 4)}%; ADP (Average Drop % - Lower is better): {round(evaluation_metrics["averaged_drop_percentage"], 4)}%;')

    # Perturbation + Deletion & Insertion tests
    for perturbation_type in [PerturbationType.POS, PerturbationType.NEG]:
        images_and_masks = read_image_and_mask_from_pickls_by_path(image_path=imagenet_val_images_folder_path,
                                                                   mask_path=pkl_path,
                                                                   device=device,
                                                                   is_explainee_convnet=is_explainee_convnet,
                                                                   )

        perturbation_config = {'perturbation_type': perturbation_type,
                               "is_calculate_deletion_insertion": True,
                               }

        print(
            f'Perturbation tests - {perturbation_config["perturbation_type"].name};  OPT_METRIC_TYPE: {opt_metric_type}')

        auc_perturbation_list, auc_deletion_insertion_list = infer_perturbation_tests(
            images_and_masks=images_and_masks,
            model_for_image_classification=model_for_classification_image,
            perturbation_config=perturbation_config,
            gt_classes_list=gt_classes_list,
            is_explainee_convnet=is_explainee_convnet,
        )
        auc_perturbation, auc_deletion_insertion = np.mean(auc_perturbation_list), np.mean(auc_deletion_insertion_list)

        print(
            f'OPT_METRIC_TYPE: {opt_metric_type}; {"Base" if is_base_model else "Opt"} + {target_or_predicted_model} Model; Perturbation tests {perturbation_config["perturbation_type"].name}, {PERTURBATION_DELETION_INSERTION_MAPPING[perturbation_config["perturbation_type"]]} test. pkl_path: {pkl_path}')
        print(
            f' OPT_METRIC_TYPE: {opt_metric_type}. Mean {perturbation_type} Perturbation AUC: {auc_perturbation}; Mean {PERTURBATION_DELETION_INSERTION_MAPPING[perturbation_config["perturbation_type"]]} AUC: {auc_deletion_insertion}')
        print('************************************************************************************')


if __name__ == '__main__':
    PERTURBATION_DELETION_INSERTION_MAPPING = {PerturbationType.POS: "Deletion", PerturbationType.NEG: "Insertion"}
    gt_classes_list = get_gt_classes(GT_VALIDATION_PATH_LABELS)

    for explainer_explainee_backbones in EXPLAINER_EXPLAINEE_BACKBONE_DETAILS.keys():
        for target_or_predicted_model in ["target", "predicted"]:
            for OPT_METRIC_TYPE in [PerturbationType.POS, PerturbationType.NEG]: # optimization type in EEA
                HOME_BASE_PATH = \
                    EXPLAINER_EXPLAINEE_BACKBONE_DETAILS[explainer_explainee_backbones]["experiment_base_path"][
                        f"{OPT_METRIC_TYPE}-opt"][
                        target_or_predicted_model]
                OPTIMIZATION_PKL_PATH = Path(HOME_BASE_PATH)
                OPTIMIZATION_PKL_PATH_BASE = Path(OPTIMIZATION_PKL_PATH, "base_model", "objects_pkl")
                OPTIMIZATION_PKL_PATH_OPT = Path(OPTIMIZATION_PKL_PATH, "opt_model", "objects_pkl")
                explainer_model_name = EXPLAINER_EXPLAINEE_BACKBONE_DETAILS[explainer_explainee_backbones][
                    "explainer"]
                explainee_model_name = EXPLAINER_EXPLAINEE_BACKBONE_DETAILS[explainer_explainee_backbones][
                    "explainee"]
                IMG_SIZE = EXPLAINER_EXPLAINEE_BACKBONE_DETAILS[explainer_explainee_backbones]["img_size"]

                EXPLAINEE_MODEL_NAME, EXPLAINER_MODEL_NAME = MODEL_ALIAS_MAPPING[explainee_model_name], \
                                                             MODEL_ALIAS_MAPPING[explainer_model_name]

                IS_EXPLANIEE_CONVNET = True if EXPLAINEE_MODEL_NAME in CONVNET_MODELS_BY_NAME.keys() else False
                IS_EXPLAINER_CONVNET = True if EXPLAINER_MODEL_NAME in CONVNET_MODELS_BY_NAME.keys() else False

                model_for_classification_image, model_for_mask_generation, feature_extractor = load_explainer_explaniee_models_and_feature_extractor(
                    explainee_model_name=EXPLAINEE_MODEL_NAME,
                    explainer_model_name=EXPLAINER_MODEL_NAME,
                    img_size=IMG_SIZE,
                    activation_function="sigmoid",
                )
                model_for_classification_image = model_for_classification_image.to(device)
                if len(os.listdir(OPTIMIZATION_PKL_PATH_BASE)) == 50000:
                    run_evaluations(pkl_path=OPTIMIZATION_PKL_PATH_BASE,
                                    is_base_model=True,
                                    target_or_predicted_model=target_or_predicted_model,
                                    backbone_name=explainer_explainee_backbones,
                                    imagenet_val_images_folder_path=IMAGENET_VAL_IMAGES_FOLDER_PATH,
                                    device=device,
                                    is_explainee_convnet=IS_EXPLANIEE_CONVNET,
                                    opt_metric_type=OPT_METRIC_TYPE.name,
                                    )
                if len(os.listdir(OPTIMIZATION_PKL_PATH_OPT)) == 50000:
                    run_evaluations(pkl_path=OPTIMIZATION_PKL_PATH_OPT,
                                    is_base_model=False,
                                    target_or_predicted_model=target_or_predicted_model,
                                    backbone_name=explainer_explainee_backbones,
                                    imagenet_val_images_folder_path=IMAGENET_VAL_IMAGES_FOLDER_PATH,
                                    device=device,
                                    is_explainee_convnet=IS_EXPLANIEE_CONVNET,
                                    opt_metric_type=OPT_METRIC_TYPE.name,
                                    )
