from pathlib import Path
from utils.consts import PEEA_CHECKPOINTS_PATH

VIT_BASE_224_VIT_BASE_224_PREDICTED_CKPT_PATH = Path(PEEA_CHECKPOINTS_PATH, "vit_base",
                                                     "pLTE_vit_base_224_predicted_best_auc__epoch=27_val_epoch_auc=0.ckpt")
VIT_BASE_224_VIT_BASE_224_TARGET_CKPT_PATH = Path(PEEA_CHECKPOINTS_PATH, "vit_base",
                                                  "pLTE_vit_base_224_target_best_auc__epoch=55_val_epoch_auc=0.ckpt")

VIT_SMALL_224_VIT_SMALL_224_PREDICTED_CKPT_PATH = Path(PEEA_CHECKPOINTS_PATH, "vit_small",
                                                       "pLTE_vit_small_224_predicted_best_auc__epoch=3_val_epoch_auc=0.ckpt")
VIT_SMALL_224_VIT_SMALL_224_TRAGET_CKPT_PATH = Path(PEEA_CHECKPOINTS_PATH, "vit_small",
                                                    "pLTE_vit_small_224_target_best_auc__epoch=11_val_epoch_auc=0.ckpt")

RESNET_RESNET_TARGET_CKPT_PATH = Path(PEEA_CHECKPOINTS_PATH, "resnet_resnet", "pEEA_resnet_resnet_target_best_auc__epoch=2_val_epoch_auc=0.ckpt")
RESNET_RESNET_PREDICTED_CKPT_PATH = Path(PEEA_CHECKPOINTS_PATH, "resnet_resnet", "pEEA_resnet_resnet_predicted_best_auc__epoch=2_val_epoch_auc=0.ckpt")
DENSENET_DENSENET_TARGET_CKPT_PATH = Path(PEEA_CHECKPOINTS_PATH, "densenet_densenet", "pEEA_densenet_densenet_target_best_auc__epoch=2_val_epoch_auc=0.ckpt")
DENSENET_DENSENET_PREDICTED_CKPT_PATH = Path(PEEA_CHECKPOINTS_PATH, "densenet_densenet", "pEEA_densenet_densenet_predicted_best_auc__epoch=3_val_epoch_auc=0.ckpt")


EXPLAINER_EXPLAINEE_BACKBONE_DETAILS = {  # key: explainer_name-explainee_name
    "vit_base_224-vit_base_224": {
        "ckpt_path": {"target": VIT_BASE_224_VIT_BASE_224_TARGET_CKPT_PATH,
                      "predicted": VIT_BASE_224_VIT_BASE_224_PREDICTED_CKPT_PATH},
        "img_size": 224,
        "patch_size": 16,
        "mask_loss": 50,
        "explainer": "vit_base_224",
        "explainee": "vit_base_224",
        "experiment_base_path": {
            "target": None,
            "predicted": None,
        },
    },
    "vit_small_224-vit_small_224": {
        "ckpt_path": {"target": VIT_SMALL_224_VIT_SMALL_224_TRAGET_CKPT_PATH,
                      "predicted": VIT_SMALL_224_VIT_SMALL_224_PREDICTED_CKPT_PATH},
        "img_size": 224,
        "patch_size": 16,
        "mask_loss": 30,
        "explainer": "vit_small_224",
        "explainee": "vit_small_224",
        "experiment_base_path": {
            "target": None,
            "predicted": None,
        },
    },
    "densenet-densenet": {
        "ckpt_path": {"target": DENSENET_DENSENET_TARGET_CKPT_PATH,
                      "predicted": DENSENET_DENSENET_PREDICTED_CKPT_PATH,
                      },
        "img_size": 224,
        "patch_size": None,
        "mask_loss": 50,
        "explainer": "densenet",
        "explainee": "densenet",
        "experiment_base_path": {
            "POS-opt": {
                "target": None,
                "predicted": None
            },
            "NEG-opt": {
                "target": None,
                "predicted": None,
            }

        },
    },
    "resnet-resnet": {
        "ckpt_path": {"target": RESNET_RESNET_TARGET_CKPT_PATH,
                      "predicted": RESNET_RESNET_PREDICTED_CKPT_PATH,
                      },
        "img_size": 224,
        "patch_size": None,
        "mask_loss": 50,
        "explainer": "resnet",
        "explainee": "resnet",
        "experiment_base_path": {
            "POS-opt": {
                "target": None,
                "predicted": None
            },
            "NEG-opt":
                {
                    "target": None,
                    "predicted": None
                }
        },
    },
}
