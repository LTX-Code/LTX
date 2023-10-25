import numpy as np
import torch
from torch import nn
# from torchvision.models import DenseNet, ResNet
from transformers import ViTForImageClassification
from feature_extractor import ViTFeatureExtractor
from main.seg_classification.backbone_to_details import EXPLAINER_EXPLAINEE_BACKBONE_DETAILS
from models.modeling_vit import ViTBasicForForImageClassification
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Tuple, NewType
from pathlib import Path
from utils.consts import IMAGES_FOLDER_PATH
from utils.transformation import image_transformations, wolf_image_transformations
from utils.utils_functions import get_image_from_path

cuda = torch.cuda.is_available()
ce_loss = nn.CrossEntropyLoss(reduction="mean")

VitModelForClassification = NewType("VitModelForClassification", ViTForImageClassification)

vit_model_types = {
    "vit-basic": ViTBasicForForImageClassification,
}


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def plot_vis_on_image(original_image,
                      mask,
                      file_name: str,
                      ):
    """
    :param original_image.shape: [3, 224, 224]
    :param mask.shape: [1,1, 224, 224]:
    """
    mask = mask.data.squeeze(0).squeeze(0).cpu().numpy()  # [1,1,224,224]
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    original_image = original_image.squeeze(0) if len(original_image.shape) == 4 else original_image
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(img=image_transformer_attribution, mask=mask)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    # plt.axis('off')
    plt.imsave(fname=Path(f"{file_name}.png"), dpi=300, arr=vis, format="png")


def visu(original_image, transformer_attribution, file_name: str, img_size: int, patch_size: int):
    """
    :param original_image: shape: [3, 224, 224]
    :param transformer_attribution: shape: [n_patches, n_patches] = [14, 14]
    :param file_name:
    :return:
    """
    if type(transformer_attribution) == np.ndarray:
        transformer_attribution = torch.tensor(transformer_attribution)
    transformer_attribution = transformer_attribution.reshape(1, int(img_size / patch_size),
                                                              int(img_size / patch_size))
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution.unsqueeze(0), scale_factor=patch_size, mode="bilinear"
    )
    transformer_attribution = transformer_attribution.reshape(img_size, img_size).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min()
    )
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (
                                            image_transformer_attribution - image_transformer_attribution.min()
                                    ) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    plt.imsave(fname=Path(f"{file_name}.png"), dpi=600, arr=vis, format="png")


def freeze_all_model_params(model: VitModelForClassification) -> VitModelForClassification:
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_x_attention_params(model: VitModelForClassification) -> VitModelForClassification:
    for param in model.named_parameters():
        if param[0] == "vit.encoder.x_attention":
            param[1].requires_grad = True
    return model


def calculate_num_of_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_num_of_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def calculate_percentage_of_trainable_params(model) -> str:
    return f"{round(calculate_num_of_trainable_params(model) / calculate_num_of_params(model), 2) * 100}%"


def print_number_of_trainable_and_not_trainable_params(model) -> None:
    print(
        f"Number of params: {calculate_num_of_params(model)}, Number of trainable params: {calculate_num_of_trainable_params(model)}"
    )


def freeze_multitask_model(model,
                           is_freezing_explaniee_model: bool = True,
                           explainer_model_n_first_layers_to_freeze: int = 0,
                           is_explainer_convnet: bool = False,
                           ):
    if is_freezing_explaniee_model:
        for param in model.vit_for_classification_image.parameters():
            param.requires_grad = False
    if is_explainer_convnet:
        pass
        """
        ct = 0
        for child in model.children():
            if ct <= explainer_model_n_first_layers_to_freeze:
                print(ct)
                for param in child.parameters():
                    param.requires_grad = False
            ct += 1
        print(print_number_of_trainable_and_not_trainable_params(model))
        """
    else:
        modules = [model.vit_for_patch_classification.vit.embeddings,
                   model.vit_for_patch_classification.vit.encoder.layer[
                   :explainer_model_n_first_layers_to_freeze]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    return model


def get_image_and_inputs_and_transformed_image(
        feature_extractor: ViTFeatureExtractor,
        image_name: str = None,
        image=None,
        is_competitive_method_transforms: bool = False,
):
    if image is None and image_name is not None:
        image = get_image_from_path(Path(IMAGES_FOLDER_PATH, image_name))
    inputs = feature_extractor(images=image, return_tensors="pt")
    transformed_image = (
        wolf_image_transformations(image) if is_competitive_method_transforms else image_transformations(image)
    )
    return inputs, transformed_image


def get_warmup_steps_and_total_training_steps(
        n_epochs: int,
        train_samples_length: int,
        batch_size: int,
) -> Tuple[int, int]:
    steps_per_epoch = train_samples_length // batch_size
    total_training_steps = steps_per_epoch * n_epochs
    warmup_steps = total_training_steps // 5
    return warmup_steps, total_training_steps


def normalize_losses(mask_loss_mul: float,
                     prediction_loss_mul: float,
                     ) -> Tuple[float, float]:
    s = mask_loss_mul + prediction_loss_mul
    mask_loss_mul_norm = mask_loss_mul / s
    pred_loss_mul_norm = prediction_loss_mul / s
    return mask_loss_mul_norm, pred_loss_mul_norm


def get_loss_multipliers(normalize: bool,
                         mask_loss_mul: int,
                         prediction_loss_mul: int,
                         ) -> Dict[str, int]:
    if normalize:
        mask_loss_mul, prediction_loss_mul = normalize_losses(mask_loss_mul=mask_loss_mul,
                                                              prediction_loss_mul=prediction_loss_mul)
    else:
        prediction_loss_mul = prediction_loss_mul
        mask_loss_mul = mask_loss_mul
    return dict(prediction_loss_mul=prediction_loss_mul, mask_loss_mul=mask_loss_mul)


def get_checkpoint_idx(ckpt_path: str) -> int:
    return int(str(ckpt_path).split("epoch=")[-1].split("_val")[0]) + 1


def get_ckpt_model_auc(ckpt_path: str) -> float:
    return float(str(ckpt_path).split("epoch_auc=")[-1].split(".ckpt")[0])


def get_params_from_config(config_vit: Dict) -> Dict:
    loss_config = config_vit["seg_cls"]["loss"]
    batch_size = config_vit["batch_size"]
    n_epochs = config_vit["n_epochs"]
    n_epochs_to_optimize_stage_b = config_vit["n_epochs_to_optimize_stage_b"]
    optimize_by_pos = config_vit["optimize_by_pos"]
    is_sampled_train_data_uniformly = config_vit["is_sampled_train_data_uniformly"]
    is_sampled_val_data_uniformly = config_vit["is_sampled_val_data_uniformly"]
    train_model_by_target_gt_class = config_vit["train_model_by_target_gt_class"]
    is_freezing_explaniee_model = config_vit["is_freezing_explaniee_model"]
    explainer_model_n_first_layers_to_freeze = config_vit["explainer_model_n_first_layers_to_freeze"]
    is_clamp_between_0_to_1 = config_vit["is_clamp_between_0_to_1"]
    enable_checkpointing = config_vit["enable_checkpointing"]
    is_competitive_method_transforms = config_vit["is_competitive_method_transforms"]
    explainer_model_name = config_vit["explainer_model_name"]
    explainee_model_name = config_vit["explainee_model_name"]
    plot_path = config_vit["plot_path"]
    default_root_dir = config_vit["default_root_dir"]
    mask_loss = loss_config["mask_loss"]
    mask_loss_mul = loss_config["mask_loss_mul"]
    prediction_loss_mul = loss_config["prediction_loss_mul"]
    lr = config_vit["lr"]
    start_epoch_to_evaluate = config_vit["start_epoch_to_evaluate"]
    n_batches_to_visualize = config_vit["n_batches_to_visualize"]
    is_ce_neg = loss_config["is_ce_neg"]
    activation_function = config_vit["activation_function"]
    RUN_BASE_MODEL = config_vit["run_base_model"]
    use_logits_only = loss_config["use_logits_only"]
    verbose = config_vit["verbose"]
    img_size = config_vit["img_size"]
    patch_size = config_vit["patch_size"]
    evaluation_experiment_folder_name = config_vit["evaluation"]["experiment_folder_name"]
    train_n_label_sample = config_vit["seg_cls"]["train_n_label_sample"]
    val_n_label_sample = config_vit["seg_cls"]["val_n_label_sample"]

    return dict(batch_size=batch_size,
                n_epochs=n_epochs,
                optimize_by_pos=optimize_by_pos,
                is_sampled_train_data_uniformly=is_sampled_train_data_uniformly,
                is_sampled_val_data_uniformly=is_sampled_val_data_uniformly,
                train_model_by_target_gt_class=train_model_by_target_gt_class,
                is_freezing_explaniee_model=is_freezing_explaniee_model,
                explainer_model_n_first_layers_to_freeze=explainer_model_n_first_layers_to_freeze,
                is_clamp_between_0_to_1=is_clamp_between_0_to_1,
                enable_checkpointing=enable_checkpointing,
                is_competitive_method_transforms=is_competitive_method_transforms,
                explainer_model_name=explainer_model_name,
                explainee_model_name=explainee_model_name,
                plot_path=plot_path,
                default_root_dir=default_root_dir,
                mask_loss=mask_loss,
                mask_loss_mul=mask_loss_mul,
                prediction_loss_mul=prediction_loss_mul,
                lr=lr,
                start_epoch_to_evaluate=start_epoch_to_evaluate,
                n_batches_to_visualize=n_batches_to_visualize,
                is_ce_neg=is_ce_neg,
                activation_function=activation_function,
                n_epochs_to_optimize_stage_b=n_epochs_to_optimize_stage_b,
                RUN_BASE_MODEL=RUN_BASE_MODEL,
                use_logits_only=use_logits_only,
                verbose=verbose,
                img_size=img_size,
                patch_size=patch_size,
                evaluation_experiment_folder_name=evaluation_experiment_folder_name,
                train_n_label_sample=train_n_label_sample,
                val_n_label_sample=val_n_label_sample,
                )


def suppress_warnings():
    import logging
    import warnings

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger('checkpoint').setLevel(0)
    logging.getLogger('lightning').setLevel(0)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


def get_backbone_details(explainer_model_name: str, explainee_model_name: str, target_or_predicted_model: str):
    EXPLAINER_EXPLAINEE_NAME = f"{explainer_model_name}-{explainee_model_name}"

    CKPT_PATH, IMG_SIZE, PATCH_SIZE, MASK_LOSS_MUL = \
        EXPLAINER_EXPLAINEE_BACKBONE_DETAILS[EXPLAINER_EXPLAINEE_NAME]["ckpt_path"][target_or_predicted_model], \
        EXPLAINER_EXPLAINEE_BACKBONE_DETAILS[EXPLAINER_EXPLAINEE_NAME]["img_size"], \
        EXPLAINER_EXPLAINEE_BACKBONE_DETAILS[EXPLAINER_EXPLAINEE_NAME]["patch_size"], \
        EXPLAINER_EXPLAINEE_BACKBONE_DETAILS[EXPLAINER_EXPLAINEE_NAME]["mask_loss"]
    CHECKPOINT_EPOCH_IDX = get_checkpoint_idx(ckpt_path=CKPT_PATH)
    BASE_CKPT_MODEL_AUC = get_ckpt_model_auc(ckpt_path=CKPT_PATH)
    return CKPT_PATH, IMG_SIZE, PATCH_SIZE, MASK_LOSS_MUL, CHECKPOINT_EPOCH_IDX, BASE_CKPT_MODEL_AUC
