import os
from distutils.util import strtobool
from icecream import ic
from cnn_baselines.evaluation.evaluation_cnn_baselines_utils import preprocess, METHOD_OPTIONS
from main.seg_classification.cnns.cnn_utils import CONVENT_NORMALIZATION_MEAN, CONVNET_NORMALIZATION_STD
from utils.vit_utils import suppress_warnings
from sklearn.metrics import auc
import torch
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt

suppress_warnings()

PERTURBATION_DELETION_INSERTION_MAPPING = {"POS": "DEL", "NEG": "INS"}


def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def eval(imagenet_ds, sample_loader, model, method: str, is_neg: bool, verbose: bool = False):
    num_samples = 0
    num_correct_model = np.zeros((len(imagenet_ds, )))
    prob_correct_model = np.zeros((len(imagenet_ds, )))
    prob_correct_model_mask = np.zeros((len(imagenet_ds, )))
    dissimilarity_model = np.zeros((len(imagenet_ds, )))
    model_index = 0

    base_size = 224 * 224
    perturbation_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    num_correct_pertub = np.zeros((9, len(imagenet_ds)))
    prob_correct_pertub = np.zeros((9, len(imagenet_ds)))
    dissimilarity_pertub = np.zeros((9, len(imagenet_ds)))
    logit_diff_pertub = np.zeros((9, len(imagenet_ds)))
    prob_diff_pertub = np.zeros((9, len(imagenet_ds)))
    perturb_index = 0
    s = 0

    for batch_idx, (data, vis, target) in enumerate(tqdm(sample_loader)):
        num_samples += len(data)
        data = data.to(device)
        vis = vis.to(device)
        target = target.to(device)

        vis_shape = vis.shape
        vis_reshaped = vis.reshape(vis_shape[0], -1)
        non_nans_indices = torch.where((torch.any(vis_reshaped.isnan(), dim=1)), 0, 1).nonzero().T[0]
        if vis.shape[0] - non_nans_indices.shape[0] > 0:
            print(f"{vis.shape[0] - non_nans_indices.shape[0]} NaNs in batch_idx: {batch_idx}")

        data = data[non_nans_indices]
        vis = vis[non_nans_indices]
        target = target[non_nans_indices]
        s += non_nans_indices.shape[0]

        norm_data = normalize(data.clone(),
                              mean=CONVENT_NORMALIZATION_MEAN,
                              std=CONVNET_NORMALIZATION_STD,
                              )

        # Compute model accuracy
        pred = model(norm_data)
        pred_probabilities = torch.softmax(pred, dim=1)
        pred_org_logit = pred.data.max(1, keepdim=True)[0].squeeze(1)
        pred_org_prob = pred_probabilities.data.max(1, keepdim=True)[0].squeeze(1)
        pred_class = pred.data.max(1, keepdim=True)[1].squeeze(1)
        tgt_pred = (target == pred_class).type(target.type()).data.cpu().numpy()
        num_correct_model[model_index:model_index + len(tgt_pred)] = tgt_pred
        if verbose:
            print(
                f'\nOriginal-top-class: {pred.data[0].argmax().item()}, pertub-top-class-logit: {round(pred_org_logit[0].item(), 4)}, Correct class logit: {round(pred.data[0][target[0]].item(), 2)} Correct class prob: {round(torch.softmax(pred.data[0], dim=0)[target[0]].item(), 5)}, Correct class: {target[0].item()}')

        # target=pred.argmax(dim=1).to(device)

        probs = torch.softmax(pred, dim=1)
        target_probs = torch.gather(probs, 1, target[:, None])[:, 0]
        prob_correct_model[model_index:model_index + len(target_probs)] = target_probs.data.cpu().numpy()

        second_probs = probs.data.topk(2, dim=1)[0][:, 1]
        temp = torch.log(target_probs / second_probs).data.cpu().numpy()
        dissimilarity_model[model_index:model_index + len(temp)] = temp

        org_shape = data.shape

        if is_neg:
            vis = -vis

        vis = vis.reshape(org_shape[0], -1)

        for i in range(len(perturbation_steps)):
            _data = data.clone()
            _, idx = torch.topk(vis, int(base_size * perturbation_steps[i]), dim=-1)  # get top k pixels
            idx = idx.unsqueeze(1).repeat(1, org_shape[1], 1)
            _data = _data.reshape(org_shape[0], org_shape[1], -1)
            _data = _data.scatter_(-1, idx, 0)
            _data = _data.reshape(*org_shape)

            _norm_data = normalize(tensor=_data,
                                   mean=CONVENT_NORMALIZATION_MEAN,
                                   std=CONVNET_NORMALIZATION_STD,
                                   )

            out = model(_norm_data)

            pred_probabilities = torch.softmax(out, dim=1)
            pred_prob = pred_probabilities.data.max(1, keepdim=True)[0].squeeze(1)
            diff = (pred_prob - pred_org_prob).data.cpu().numpy()
            prob_diff_pertub[i, perturb_index:perturb_index + len(diff)] = diff

            pred_logit = out.data.max(1, keepdim=True)[0].squeeze(1)
            diff = (pred_logit - pred_org_logit).data.cpu().numpy()
            logit_diff_pertub[i, perturb_index:perturb_index + len(diff)] = diff

            target_class = out.data.max(1, keepdim=True)[1].squeeze(1)
            temp = (target == target_class).type(target.type()).data.cpu().numpy()
            num_correct_pertub[i, perturb_index:perturb_index + len(temp)] = temp

            if verbose:
                # for image_idx in range(2):
                if i < 3:
                    plot_image(image=_data[0])

                print(
                    f'{100 * perturbation_steps[i]}% pixels blacked. pertub-top-class: {out.data[0].argmax().item()}, pertub-top-class-logit: {round(pred_logit[0].item(), 4)}, Correct class logit: {round(out.data[0][target[0]].item(), 2)} Correct class prob: {round(pred_probabilities[0][target[0]].item(), 5)}, Correct class: {target[0].item()}')

            probs_pertub = torch.softmax(out, dim=1)
            target_probs = torch.gather(probs_pertub, 1, target[:, None])[:, 0]
            prob_correct_pertub[i, perturb_index:perturb_index + len(temp)] = target_probs.data.cpu().numpy()

            second_probs = probs_pertub.data.topk(2, dim=1)[0][:, 1]
            temp = torch.log(target_probs / second_probs).data.cpu().numpy()
            dissimilarity_pertub[i, perturb_index:perturb_index + len(temp)] = temp

        model_index += len(target)
        perturb_index += len(target)

    init_mean_accuracy = np.mean(num_correct_model)
    values = np.mean(num_correct_pertub, axis=1)
    values = np.insert(values, 0, init_mean_accuracy)
    steps = np.arange(0, 1, 0.1)
    auc_score = auc(steps, values) * 100
    # print(f"Pertubation Step Avg. : {np.mean(num_correct_pertub, axis=1)}")

    print(f"total_images after nans removal: {s}")
    print(f'AUC  = {auc_score} \n')
    init_mean_accuracy = np.mean(prob_correct_model)
    values = np.mean(prob_correct_pertub, axis=1)
    values = np.insert(values, 0, init_mean_accuracy)
    steps = np.arange(0, 1, 0.1)
    auc_score = auc(steps, values) * 100
    print(f"Proba AUC - {PERTURBATION_DELETION_INSERTION_MAPPING['NEG' if is_neg else 'POS']}: {auc_score} \n\n\n")
    # print(
    #     f'np.mean(num_correct_model)= {np.mean(num_correct_model)} \n np.std(num_correct_model) = {np.std(num_correct_model)}')
    # print(
    #     f'np.mean(dissimilarity_model)= {np.mean(dissimilarity_model)} \n np.std(dissimilarity_model) = {np.std(dissimilarity_model)}')
    # print(
    #     f'np.mean(num_correct_pertub)= {np.mean(num_correct_pertub, axis=1)} \n np.std(num_correct_pertub) = {np.std(num_correct_pertub, axis=1)}')
    # print(
    #     f'np.mean(dissimilarity_pertub)= {np.mean(dissimilarity_pertub, axis=1)} \n np.std(dissimilarity_pertub) = {np.std(dissimilarity_pertub, axis=1)}')


def plot_image(image, title: str = None):
    plt.imshow(image.cpu().detach().permute(1, 2, 0).numpy())
    plt.axis('off');
    if title is not None:
        plt.title(title)
    plt.show();


def show_mask(mask):
    plt.imshow(mask[0].cpu().detach())
    plt.axis('off');
    plt.show();


def calculate_auc(mean_accuracy_by_step: np.ndarray) -> float:
    return auc(x=np.arange(0, 1, 0.1), y=mean_accuracy_by_step)


def get_auc(num_correct_pertub, num_correct_model):
    """
    num_correct_pertub is matrix of each row represents perurbation step. Each column represents masked image
    Each cell represents if the prediction at that step is the right prediction (0 / 1) and average of the images axis to
    get the number of average correct prediction at each perturbation step and then trapz integral (auc) to get final val
    """
    mean_accuracy_by_step = np.mean(num_correct_pertub, axis=1)
    # mean_accuracy_by_step = np.insert(mean_accuracy_by_step, 0,
    #                                   1)  # TODO - accuracy for class. Now its top-class (predicted)
    mean_accuracy_by_step = np.insert(mean_accuracy_by_step, 0, np.mean(num_correct_model))
    auc = calculate_auc(mean_accuracy_by_step=mean_accuracy_by_step) * 100
    # print(num_correct_pertub)
    print(f'AUC: {round(auc, 4)}% for {num_correct_pertub.shape[1]} records')
    return auc


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=./:$PYTHONPATH nohup python cnn_baselines/evaluation/perturbation_eval_from_hdf5_all_comb_by_method.py --method gradcam &> nohups_logs/journal/cnn_baselines/eval/pertub_gradcam_all_combinations.out &
    """
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    parser = argparse.ArgumentParser(description="Perturbation Evaluation")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help='',
                        )
    parser.add_argument("--neg",
                        type=lambda x: bool(strtobool(x)),
                        nargs="?",
                        const=True,
                        default=False)
    parser.add_argument("--backbone", type=str,
                        default="resnet101",
                        choices=["resnet101", "densenet"],
                        )
    parser.add_argument("--method",
                        type=str,
                        default="ig",
                        choices=METHOD_OPTIONS,
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

    # print(args)
    #############################################################################
    args.backbone = "resnet101"
    args.neg = False
    args.is_target = True
    print(args)
    imagenet_ds, sample_loader, model = preprocess(backbone=args.backbone,
                                                   method=args.method,
                                                   is_target=args.is_target,
                                                   batch_size=args.batch_size,
                                                   )
    eval(imagenet_ds=imagenet_ds,
         sample_loader=sample_loader,
         model=model,
         method=args.method,
         is_neg=args.neg,
         verbose=args.verbose,
         )
    torch.cuda.empty_cache()
    #############################################################################
    args.backbone = "resnet101"
    args.neg = True
    args.is_target = True
    print(args)
    imagenet_ds, sample_loader, model = preprocess(backbone=args.backbone,
                                                   method=args.method,
                                                   is_target=args.is_target,
                                                   batch_size=args.batch_size,
                                                   )
    eval(imagenet_ds=imagenet_ds,
         sample_loader=sample_loader,
         model=model,
         method=args.method,
         is_neg=args.neg,
         verbose=args.verbose,
         )
    torch.cuda.empty_cache()
    ############################################################################
    args.backbone = "resnet101"
    args.neg = False
    args.is_target = False
    print(args)
    imagenet_ds, sample_loader, model = preprocess(backbone=args.backbone,
                                                   method=args.method,
                                                   is_target=args.is_target,
                                                   batch_size=args.batch_size,
                                                   )
    eval(imagenet_ds=imagenet_ds,
         sample_loader=sample_loader,
         model=model,
         method=args.method,
         is_neg=args.neg,
         verbose=args.verbose,
         )
    torch.cuda.empty_cache()
    #############################################################################
    args.backbone = "resnet101"
    args.neg = True
    args.is_target = False
    print(args)
    imagenet_ds, sample_loader, model = preprocess(backbone=args.backbone,
                                                   method=args.method,
                                                   is_target=args.is_target,
                                                   batch_size=args.batch_size,
                                                   )
    eval(imagenet_ds=imagenet_ds,
         sample_loader=sample_loader,
         model=model,
         method=args.method,
         is_neg=args.neg,
         verbose=args.verbose,
         )
    torch.cuda.empty_cache()
    #############################################################################
    args.backbone = "densenet"
    args.neg = False
    args.is_target = True
    print(args)
    imagenet_ds, sample_loader, model = preprocess(backbone=args.backbone,
                                                   method=args.method,
                                                   is_target=args.is_target,
                                                   batch_size=args.batch_size,
                                                   )
    eval(imagenet_ds=imagenet_ds,
         sample_loader=sample_loader,
         model=model,
         method=args.method,
         is_neg=args.neg,
         verbose=args.verbose,
         )
    torch.cuda.empty_cache()
    #############################################################################
    args.backbone = "densenet"
    args.neg = True
    args.is_target = True
    print(args)
    imagenet_ds, sample_loader, model = preprocess(backbone=args.backbone,
                                                   method=args.method,
                                                   is_target=args.is_target,
                                                   batch_size=args.batch_size,
                                                   )
    eval(imagenet_ds=imagenet_ds,
         sample_loader=sample_loader,
         model=model,
         method=args.method,
         is_neg=args.neg,
         verbose=args.verbose,
         )
    torch.cuda.empty_cache()
    ############################################################################
    args.backbone = "densenet"
    args.neg = False
    args.is_target = False
    print(args)
    imagenet_ds, sample_loader, model = preprocess(backbone=args.backbone,
                                                   method=args.method,
                                                   is_target=args.is_target,
                                                   batch_size=args.batch_size,
                                                   )
    eval(imagenet_ds=imagenet_ds,
         sample_loader=sample_loader,
         model=model,
         method=args.method,
         is_neg=args.neg,
         verbose=args.verbose,
         )
    torch.cuda.empty_cache()
    #############################################################################
    args.backbone = "densenet"
    args.neg = True
    args.is_target = False
    print(args)
    imagenet_ds, sample_loader, model = preprocess(backbone=args.backbone,
                                                   method=args.method,
                                                   is_target=args.is_target,
                                                   batch_size=args.batch_size,
                                                   )
    eval(imagenet_ds=imagenet_ds,
         sample_loader=sample_loader,
         model=model,
         method=args.method,
         is_neg=args.neg,
         verbose=args.verbose,
         )
    torch.cuda.empty_cache()
