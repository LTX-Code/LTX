from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import models
from cnn_baselines.evaluation.imangenet_results_cnn_baselines import ImagenetResults
from utils.consts import CNN_BASELINES_RESULTS_PATH

METHOD_OPTIONS = ['lift-cam', 'layercam', 'ig', 'ablation-cam', 'fullgrad', 'gradcam', 'gradcampp']
FEATURE_LAYER_NUMBER_BY_BACKBONE = {'resnet101': 8, 'densenet': 12}


def preprocess(backbone: str, method: str, is_target: bool, batch_size: int):
    runs_dir = Path(CNN_BASELINES_RESULTS_PATH, "visualizations", backbone, method,
                    "target" if is_target else "predicted")
    print(runs_dir)
    imagenet_ds = ImagenetResults(runs_dir)
    if backbone == "resnet101":
        model = models.resnet101(pretrained=True).cuda()
    elif backbone == "densenet":
        model = models.densenet201(pretrained=True).cuda()
    else:
        raise ("Backbone not implemented")
    model.eval()
    sample_loader = DataLoader(
        imagenet_ds,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
    )
    return imagenet_ds, sample_loader, model
