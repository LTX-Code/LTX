import os
from distutils.util import strtobool
import argparse
from torch.utils.data import DataLoader
from cnn_baselines.evaluation.evaluation_cnn_baselines_utils import METHOD_OPTIONS, FEATURE_LAYER_NUMBER_BY_BACKBONE
from cnn_baselines.imagenet_dataset_cnn_baselines import ImageNetDataset
from main.seg_classification.cnns.cnn_utils import convnet_preprocess, convnet_resize_transform
from pathlib import Path
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from cnn_baselines.grad_methods_utils import run_by_class_grad
from cnn_baselines.saliency_models import GradModel, ReLU, lift_cam, ig_captum, generic_torchcam
from utils.consts import IMAGENET_VAL_IMAGES_FOLDER_PATH, CNN_BASELINES_RESULTS_PATH
from cnn_baselines.fullgrad_method.fullgrad import FullGrad
from cnn_baselines.torchgc.pytorch_grad_cam.layer_cam import LayerCAM
from cnn_baselines.torchgc.pytorch_grad_cam.score_cam import ScoreCAM
from cnn_baselines.torchgc.pytorch_grad_cam.ablation_cam import AblationCAM
import h5py
import numpy as np
from icecream import ic
import gc
device = torch.device('cuda')
USE_MASK = True


def show_mask(mask, title: str = None):
    plt.imshow(mask[0].cpu().detach())
    if title is not None:
        plt.title(title);
    plt.axis('off');
    plt.show();


def compute_saliency_and_save(dir: Path,
                              model,
                              method: str,
                              dataloader: DataLoader,
                              vis_class: str,
                              backbone_name: str,
                              verbose: bool,
                              ):
    first = True
    with h5py.File(os.path.join(dir, 'results.hdf5'), 'a') as f:
        data_cam = f.create_dataset('vis',
                                    (1, 1, 224, 224),
                                    maxshape=(None, 1, 224, 224),
                                    dtype=np.float32,
                                    compression="gzip")
        data_image = f.create_dataset('image',
                                      (1, 3, 224, 224),
                                      maxshape=(None, 3, 224, 224),
                                      dtype=np.float32,
                                      compression="gzip")
        data_target = f.create_dataset('target',
                                       (1,),
                                       maxshape=(None,),
                                       dtype=np.int32,
                                       compression="gzip")

        for batch_idx, (data, target, resized_image) in enumerate(tqdm(dataloader)):
            torch.cuda.empty_cache()
            gc.collect()
            if first:
                first = False
                data_cam.resize(data_cam.shape[0] + data.shape[0] - 1, axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0] - 1, axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0] - 1, axis=0)
            else:
                data_cam.resize(data_cam.shape[0] + data.shape[0], axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0], axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0], axis=0)

            # Add data
            data_image[-data.shape[0]:] = resized_image.data.cpu().numpy()
            data_target[-data.shape[0]:] = target.data.cpu().numpy()

            data = data.to(device)
            data.requires_grad_()

            input_predictions = model(data.to(device), hook=False).detach()
            predicted_label = torch.argmax(input_predictions, dim=1)

            index = predicted_label
            if vis_class == 'target':
                index = target
            index = index.to(device)

            if method == 'lift-cam':
                heatmap = lift_cam(
                    model=model,
                    inputs=data,
                    label=index,
                    device=device,
                    use_mask=USE_MASK,
                )

            elif method == 'score-cam':
                heatmap = generic_torchcam(
                    modelCAM=ScoreCAM,
                    backbone_name=backbone_name,
                    inputs=data,
                    label=index,
                    device=device,
                    use_mask=USE_MASK,
                )

            elif method == 'ablation-cam':
                heatmap = generic_torchcam(
                    modelCAM=AblationCAM,
                    backbone_name=backbone_name,
                    inputs=data,
                    label=index,
                    device=device,
                    use_mask=USE_MASK,
                )

            elif method == 'ig':
                heatmap = ig_captum(
                    model=model,
                    inputs=data,
                    label=index,
                    device=device,
                    use_mask=USE_MASK,
                )

            elif method == 'layercam':
                heatmap = generic_torchcam(
                    modelCAM=LayerCAM,
                    backbone_name=backbone_name,
                    inputs=data,
                    label=index,
                    device=device,
                    use_mask=USE_MASK,
                )

            elif method == 'fullgrad':
                heatmap = FullGrad(model).saliency(data)  # [bs, 1, 224, 224]
                heatmap = heatmap.cpu().detach()

            elif method in ['gradcam', 'gradcampp']:
                heatmap = run_by_class_grad(model=model,
                                            image_preprocessed=data.squeeze(0),
                                            label=index,
                                            backbone_name=backbone_name,
                                            device=device,
                                            features_layer=FEATURE_LAYER_NUMBER,
                                            method=method,
                                            use_mask=USE_MASK,
                                            )
            else:
                raise NotImplementedError

            data_cam[-data.shape[0]:] = heatmap
            if verbose:
                show_mask(heatmap, title=f"{method}-{vis_class}")


if __name__ == '__main__':
    """"
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./:$PYTHONPATH nohup python cnn_baselines/generate_visualization_cnn_baselines.py --method gradcam --backbone-name resnet101 --vis-by-target-gt-class True &> nohups_logs/journal/cnn_baselines/resnet_gradcam_target.out &
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./:$PYTHONPATH nohup python cnn_baselines/generate_visualization_cnn_baselines.py --method gradcam --backbone-name resnet101 --vis-by-target-gt-class False &> nohups_logs/journal/cnn_baselines/resnet_gradcam_predicted.out &
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=./:$PYTHONPATH nohup python cnn_baselines/generate_visualization_cnn_baselines.py --method gradcampp --backbone-name resnet101 --vis-by-target-gt-class True &> nohups_logs/journal/cnn_baselines/resnet_gradcampp_target.out &
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=./:$PYTHONPATH nohup python cnn_baselines/generate_visualization_cnn_baselines.py --method gradcampp --backbone-name resnet101 --vis-by-target-gt-class False &> nohups_logs/journal/cnn_baselines/resnet_gradcampp_predicted.out &
    """
    parser = argparse.ArgumentParser(description='Generate CNN baselines visualizations')
    parser.add_argument('--method', type=str, default="gradcam", choices=METHOD_OPTIONS)
    parser.add_argument('--backbone-name',
                        type=str,
                        default="resnet101",
                        choices=list(FEATURE_LAYER_NUMBER_BY_BACKBONE.keys()),
                        )
    parser.add_argument("--vis-by-target-gt-class",
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        )

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument("--verbose",
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=False,
                        )

    args = parser.parse_args()
    args.batch_size = 8 if args.method == "fullgrad" else args.batch_size

    ic(args.method)
    ic(args.backbone_name)
    ic(args.vis_by_target_gt_class)
    ic(args.verbose)
    ic(args.batch_size)

    vis_class = "target" if args.vis_by_target_gt_class else "predicted"

    FEATURE_LAYER_NUMBER = FEATURE_LAYER_NUMBER_BY_BACKBONE[args.backbone_name]
    PREV_LAYER = FEATURE_LAYER_NUMBER - 1

    torch.nn.modules.activation.ReLU.forward = ReLU.forward

    model = GradModel(args.backbone_name, feature_layer=FEATURE_LAYER_NUMBER)
    model.to(device)
    model.eval()
    model.zero_grad()

    BASE_FOLDER_NAME = "visualizations"
    os.makedirs(Path(CNN_BASELINES_RESULTS_PATH, BASE_FOLDER_NAME), exist_ok=True)
    dir_path = Path(CNN_BASELINES_RESULTS_PATH, f'{BASE_FOLDER_NAME}/{args.backbone_name}/{args.method}/{vis_class}')
    os.makedirs(dir_path, exist_ok=True)

    print(dir_path)

    imagenet_ds = ImageNetDataset(root_dir=IMAGENET_VAL_IMAGES_FOLDER_PATH,
                                  transform=convnet_preprocess,
                                  resize_transform=convnet_resize_transform,
                                  )

    sample_loader = DataLoader(
        imagenet_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    compute_saliency_and_save(dir=dir_path,
                              model=model,
                              method=args.method,
                              dataloader=sample_loader,
                              vis_class=vis_class,
                              backbone_name=args.backbone_name,
                              verbose=args.verbose,
                              )
