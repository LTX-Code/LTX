import os
from distutils.util import strtobool
from cnn_baselines.evaluation.evaluation_cnn_baselines_utils import preprocess, METHOD_OPTIONS
from utils.vit_utils import suppress_warnings
from main.seg_classification.cnns.cnn_utils import CONVENT_NORMALIZATION_MEAN, CONVNET_NORMALIZATION_STD
import torch
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt
from icecream import ic

suppress_warnings()


def show_plots(n_images: int, method: str, vis, img_with_mask, target_probs, target_probs_mask) -> None:
    for image_idx in range(n_images):
        show_mask(vis[image_idx])
        plot_image(img_with_mask[image_idx],
                   title=f"{method} - Original:{round(target_probs[image_idx].item(), 3)}, masked: {round(target_probs_mask[image_idx].item(), 3)}")


def normalize(tensor,
              mean=[0.5, 0.5, 0.5],
              std=[0.5, 0.5, 0.5],
              ):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def eval(imagenet_ds, sample_loader, model, method: str, verbose: bool = False):
    prob_correct_model = np.zeros((len(imagenet_ds, )))
    prob_correct_model_mask = np.zeros((len(imagenet_ds, )))
    model_index = 0

    for batch_idx, (data, vis, target) in enumerate(tqdm(sample_loader)):
        data = data.to(device)
        # plot_image(data[0])

        vis = vis.to(device)
        # show_mask(vis[0])
        # show_mask(data[0] * vis[0])

        target = target.to(device)
        norm_data = normalize(data.clone(),
                              mean=CONVENT_NORMALIZATION_MEAN,
                              std=CONVNET_NORMALIZATION_STD,
                              )

        # Compute model accuracy
        pred = model(norm_data)
        probs = torch.softmax(pred, dim=1)
        target_probs = torch.gather(probs, 1, target[:, None])[:, 0]
        # proba of original image
        prob_correct_model[model_index:model_index + len(target_probs)] = target_probs.data.cpu().numpy()

        # #### ADP PIC
        img_with_mask = data.clone() * vis.clone()
        norm_data_mask = normalize(img_with_mask.clone(),
                                   mean=CONVENT_NORMALIZATION_MEAN,
                                   std=CONVNET_NORMALIZATION_STD,
                                   )
        pred_mask = model(norm_data_mask)
        probs_mask = torch.softmax(pred_mask, dim=1)
        target_probs_mask = torch.gather(probs_mask, 1, target[:, None])[:, 0]
        if verbose:
            show_plots(n_images=20,
                       method=method,
                       vis=vis,
                       img_with_mask=img_with_mask,
                       target_probs=target_probs,
                       target_probs_mask=target_probs_mask,
                       )
        prob_correct_model_mask[model_index:model_index + len(target_probs)] = target_probs_mask.data.cpu().numpy()

        model_index += len(target)
        if torch.isnan(vis).any():
            print(f"batch_idx: {batch_idx} contains nan values")

    x = torch.tensor(prob_correct_model)
    y = torch.tensor(prob_correct_model_mask)
    non_nans_indices = torch.where((torch.isnan(y)), 0, 1).nonzero().T[0]
    ic(len(non_nans_indices))
    x = x[non_nans_indices]
    y = y[non_nans_indices]

    adp = (torch.maximum(x - y, torch.zeros_like(x)) / x).mean() * 100
    pic = torch.where(x < y, 1.0, 0.0).mean() * 100
    print(f"PIC = {pic.item()}")
    print(f"ADP = {adp.item()}")


def plot_image(image, title: str = None):
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    plt.axis('off');
    if title is not None:
        plt.title(title)
    plt.show();


def show_mask(mask):
    plt.imshow(mask[0].cpu().detach())
    plt.axis('off');
    plt.show();


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./:$PYTHONPATH nohup python cnn_baselines/evaluation/adp_pic_eval_from_hdf5.py --method lift-cam --backbone resnet101 --is-target True &> nohups_logs/journal/cnn_baselines/eval/liftcam_resnet_target.out &
    """
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    parser = argparse.ArgumentParser(description="Infer from HDF5 file")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        )
    parser.add_argument("--method", type=str,
                        default="gradcam",
                        choices=METHOD_OPTIONS,
                        )
    parser.add_argument("--backbone", type=str,
                        default="resnet101",
                        choices=["resnet101", "densenet"],
                        )
    parser.add_argument("--is-target",
                        type=lambda x: bool(strtobool(x)),
                        nargs="?",
                        const=True,
                        default=True,
                        )
    parser.add_argument("--verbose",
                        type=lambda x: bool(strtobool(x)),
                        nargs="?",
                        const=True,
                        default=False,
                        )

    args = parser.parse_args()
    print(args)
    torch.multiprocessing.set_start_method("spawn")
    imagenet_ds, sample_loader, model = preprocess(backbone=args.backbone,
                                                   method=args.method,
                                                   is_target=args.is_target,
                                                   batch_size=args.batch_size,
                                                   )
    eval(imagenet_ds=imagenet_ds,
         sample_loader=sample_loader,
         model=model,
         method=args.method,
         verbose=args.verbose,
         )
