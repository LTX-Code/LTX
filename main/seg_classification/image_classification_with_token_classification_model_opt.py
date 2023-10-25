import os
import numpy as np
from pathlib import Path
from typing import Union
import pytorch_lightning as pl
from torchvision.models import DenseNet, ResNet
from config import config
from evaluation.perturbation_tests.seg_cls_perturbation_tests import (save_best_auc_objects_to_disk,
                                                                      run_perturbation_test_opt)
from main.seg_classification.image_classification_with_token_classification_model import \
    ImageClassificationWithTokenClassificationModel
from main.seg_classification.seg_cls_consts import NEG_AUC_STOP_VALUE, POS_AUC_STOP_VALUE
from models.modeling_cnn_for_mask_generation import CNNForMaskGeneration
from utils.vit_utils import visu
from models.modeling_vit_patch_classification import ViTForMaskGeneration
from transformers import ViTForImageClassification

pl.seed_everything(config["general"]["seed"])


class OptImageClassificationWithTokenClassificationModel(ImageClassificationWithTokenClassificationModel):
    def __init__(
            self,
            model_for_classification_image: Union[ViTForImageClassification, ResNet, DenseNet],
            model_for_mask_generation: Union[ViTForMaskGeneration, CNNForMaskGeneration],
            warmup_steps: int,
            total_training_steps: int,
            plot_path,
            best_auc_objects_path: str,
            best_auc_plot_path: str,
            checkpoint_epoch_idx: int,
            is_explainer_convnet: bool,
            is_explainee_convnet: bool,
            lr: float,
            n_epochs: int,
            activation_function: str,
            train_model_by_target_gt_class: bool,
            use_logits_only: bool,
            img_size: int,
            patch_size: int,
            mask_loss: str,
            mask_loss_mul: int,
            prediction_loss_mul: int,
            is_ce_neg: bool = False,
            n_batches_to_visualize: int = 2,
            start_epoch_to_evaluate: int = 1,
            is_clamp_between_0_to_1: bool = True,
            run_base_model_only: bool = False,
            verbose: bool = False,
            optimize_by_pos: bool = True,
    ):
        super().__init__(model_for_classification_image=model_for_classification_image,
                         model_for_mask_generation=model_for_mask_generation,
                         warmup_steps=warmup_steps,
                         total_training_steps=total_training_steps,
                         is_explainer_convnet=is_explainer_convnet,
                         is_explainee_convnet=is_explainee_convnet,
                         lr=lr,
                         start_epoch_to_evaluate=start_epoch_to_evaluate,
                         n_batches_to_visualize=n_batches_to_visualize,
                         activation_function=activation_function,
                         train_model_by_target_gt_class=train_model_by_target_gt_class,
                         use_logits_only=use_logits_only,
                         img_size=img_size,
                         patch_size=patch_size,
                         mask_loss=mask_loss,
                         mask_loss_mul=mask_loss_mul,
                         prediction_loss_mul=prediction_loss_mul,
                         is_ce_neg=is_ce_neg,
                         plot_path=plot_path,
                         is_clamp_between_0_to_1=is_clamp_between_0_to_1,
                         experiment_path=Path(""),
                         verbose=verbose,
                         )
        self.n_epochs = n_epochs
        self.optimize_by_pos = optimize_by_pos
        self.best_auc_objects_path = best_auc_objects_path
        self.best_auc_plot_path = best_auc_plot_path
        self.best_auc = None
        self.best_auc_epoch = None
        self.best_auc_vis = None
        self.checkpoint_epoch_idx = checkpoint_epoch_idx
        self.image_idx = None
        self.auc_by_epoch = None
        self.run_base_model_only = run_base_model_only
        self.vit_for_classification_image.eval()

    def init_auc(self) -> None:
        self.best_auc = np.inf if self.optimize_by_pos else -np.inf
        self.best_auc_epoch = 0
        self.best_auc_vis = None
        self.auc_by_epoch = []
        self.image_idx = len(os.listdir(self.best_auc_objects_path))
        self.perturbation_type = "POS" if self.optimize_by_pos else "NEG"


    def training_step(self, batch, batch_idx):
        self.vit_for_classification_image.eval()
        self.vit_for_patch_classification.encoder.eval() if self.is_explainer_convnet else self.vit_for_patch_classification.eval()
        inputs = batch["pixel_values"].squeeze(1)
        resized_and_normalized_image = batch["resized_and_normalized_image"]
        image_resized = batch["image"]
        target_class = batch["target_class"]

        if self.current_epoch == self.checkpoint_epoch_idx:
            self.init_auc()
        output = self.forward(inputs=inputs, image_resized=image_resized, target_class=target_class)
        images_mask = output.interpolated_mask

        return {
            "loss": output.lossloss_output.loss,
            "pred_loss": output.lossloss_output.pred_loss,
            "pred_loss_mul": output.lossloss_output.prediction_loss_multiplied,
            "mask_loss": output.lossloss_output.mask_loss,
            "mask_loss_mul": output.lossloss_output.mask_loss_multiplied,
            "resized_and_normalized_image": resized_and_normalized_image,
            "target_class": target_class,
            "image_mask": images_mask,
            "image_resized": image_resized,
            "patches_mask": output.tokens_mask,
            "auc": self.best_auc
        }

    def validation_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        auc = run_perturbation_test_opt(
            model=self.vit_for_classification_image,
            outputs=outputs,
            stage="train",
            epoch_idx=self.current_epoch,
            is_convnet=self.is_explainee_convnet,
            verbose=self.verbose,
            img_size=self.img_size,
            perturbation_type=self.perturbation_type,
        )
        if self.best_auc is None or (auc < self.best_auc if self.optimize_by_pos else auc > self.best_auc):
            self.best_auc = auc
            self.best_auc_epoch = self.current_epoch
            self.best_auc_vis = outputs[0]["image_mask"]
            self.best_auc_image = outputs[0]["image_resized"]

            save_best_auc_objects_to_disk(path=Path(f"{self.best_auc_objects_path}", f"{str(self.image_idx)}.pkl"),
                                          auc=auc,
                                          vis=self.best_auc_vis,
                                          original_image=self.best_auc_image,
                                          epoch_idx=self.current_epoch,
                                          )
            if self.run_base_model_only or (auc < POS_AUC_STOP_VALUE if self.optimize_by_pos else auc > NEG_AUC_STOP_VALUE):
                outputs[0]['auc'] = auc
                self.trainer.should_stop = True

        if self.current_epoch == self.n_epochs - 1:
            self.trainer.should_stop = True

    def validation_epoch_end(self, outputs):
        pass

    def visualize_images_by_outputs(self, outputs):
        image = outputs[0]["resized_and_normalized_image"].detach().cpu()
        mask = outputs[0]["patches_mask"].detach().cpu()
        auc = outputs[0]['auc']
        image = image if len(image.shape) == 3 else image.squeeze(0)
        mask = mask if len(mask.shape) == 3 else mask.squeeze(0)
        visu(
            original_image=image,
            transformer_attribution=mask,
            file_name=Path(self.best_auc_plot_path,
                           f"{str(self.image_idx)}__{self.current_epoch}__AUC_{round(auc, 0)}").resolve(),
            img_size=self.img_size,
            patch_size=self.patch_size,
        )
