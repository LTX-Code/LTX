import wandb
from typing import Dict
from config import config


def configure_log(config_vit: Dict, experiment_name: str, image_name: str):
    print(experiment_name)
    wandb.config = {
        "learning_rate": config_vit["lr"],
        "experiment_name": experiment_name,
        "image_name": image_name,
    }
    run = wandb.init(project="vit-sigmoid-1", entity=config["general"]["wandb_entity"], config=wandb.config)
    return run


def get_wandb_config(config_vit: Dict, experiment_name: str, image_name: str):
    wandb.config = {
        "learning_rate": config_vit["lr"],
        "experiment_name": experiment_name,
        "image_name": image_name,
    }
    return wandb.config
