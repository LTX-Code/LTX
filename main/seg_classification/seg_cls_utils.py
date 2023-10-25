import os
import pickle
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml
from torch import Tensor, nn
from config import config
from main.seg_classification.seg_cls_consts import OBT_OBJECTS_PLOT_FOLDER_NAME, OBT_OBJECTS_FOLDER_NAME

vit_config = config["vit"]

bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction="mean")
ce_loss = nn.CrossEntropyLoss(reduction="mean")


def l1_loss(tokens_mask) -> Tensor:
    return torch.abs(tokens_mask).mean()


def prediction_loss(output,
                    target,
                    target_class,
                    train_model_by_target_gt_class: bool,
                    use_logits_only: bool):
    if train_model_by_target_gt_class:
        target_class_to_compare = target_class
    else:
        target_class_to_compare = torch.argmax(target, dim=1)
    if use_logits_only:
        return -torch.gather(output, 1, target_class_to_compare.unsqueeze(1)).squeeze(1).mean()
    return ce_loss(output, target_class_to_compare)  # maximize the pred to original model


def encourage_token_mask_to_prior_loss(tokens_mask: Tensor, prior: int = 0):
    if prior == 0:
        target_encourage_patches = torch.zeros_like(tokens_mask)
    elif prior == 1:
        target_encourage_patches = torch.ones_like(tokens_mask)
    else:
        raise NotImplementedError
    bce_encourage_prior_patches_loss = bce_with_logits_loss(
        tokens_mask, target_encourage_patches
    )
    return bce_encourage_prior_patches_loss


def load_obj(path) -> Any:
    with open(Path(path), 'rb') as f:
        return pickle.load(f)


def load_pickles_and_calculate_auc(path):
    aucs = []
    listdir = sorted(list(Path(path).iterdir()))
    for pkl_path in listdir:
        loaded_obj = load_obj(pkl_path)
        auc = loaded_obj['auc']
        aucs.append(auc)
    print(f"{len(aucs)} samples")
    return np.mean(aucs)


def create_folder_hierarchy(base_auc_objects_path: str, exp_name: str):
    best_auc_plot_path = Path(base_auc_objects_path, exp_name, 'opt_model', OBT_OBJECTS_PLOT_FOLDER_NAME)
    best_auc_objects_path = Path(base_auc_objects_path, exp_name, 'opt_model', OBT_OBJECTS_FOLDER_NAME)
    os.makedirs(best_auc_plot_path, exist_ok=True)
    os.makedirs(best_auc_objects_path, exist_ok=True)

    base_model_best_auc_plot_path = Path(base_auc_objects_path, exp_name, 'base_model', OBT_OBJECTS_PLOT_FOLDER_NAME)
    base_model_best_auc_objects_path = Path(base_auc_objects_path, exp_name, 'base_model', OBT_OBJECTS_FOLDER_NAME)
    os.makedirs(base_model_best_auc_plot_path, exist_ok=True)
    os.makedirs(base_model_best_auc_objects_path, exist_ok=True)
    return best_auc_plot_path, best_auc_objects_path, base_model_best_auc_plot_path, base_model_best_auc_objects_path


def save_config_to_root_dir(exp_name):
    path_dir = os.path.join(vit_config["default_root_dir"], f"{exp_name}")
    os.makedirs(path_dir, exist_ok=True)
    with open(os.path.join(path_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


def get_gt_classes(path):
    with open(path, 'r') as f:
        gt_classes_list = f.readlines()
    gt_classes_list = [int(record.split()[-1].replace('\n', '')) for record in gt_classes_list]
    return gt_classes_list
