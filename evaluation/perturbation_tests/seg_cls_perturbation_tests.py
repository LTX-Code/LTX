from typing import Union, Tuple, List, Dict
import pandas as pd
from evaluation.evaluation_utils import normalize, calculate_auc
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import os
from transformers.modeling_outputs import ImageClassifierOutput
from torch import Tensor
import numpy as np
from config import config
from main.seg_classification.cnns.cnn_utils import CONVENT_NORMALIZATION_MEAN, CONVNET_NORMALIZATION_STD

vit_config = config['vit']
EXPERIMENTS_FOLDER_PATH = vit_config["experiments_path"]

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def eval_perturbation_test(experiment_dir: Path,
                           model,
                           outputs: List[Dict],
                           img_size: int = 224,
                           perturbation_type: str = "POS",
                           is_calculate_deletion_insertion: bool = False,
                           is_convenet: bool = False,
                           verbose: bool = False,
                           ) -> Union[float, Tuple[float, float]]:
    n_samples = sum(output["image_resized"].shape[0] for output in outputs)
    num_correct_model = np.zeros((n_samples))
    prob_correct_model = np.zeros((n_samples))
    model_index = 0

    base_size = img_size * img_size
    perturbation_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    num_correct_pertub = np.zeros((len(perturbation_steps), n_samples))  # 9 is the num perturbation steps
    prob_pertub = np.zeros((len(perturbation_steps), n_samples))
    perturb_index = 0
    for batch in outputs:
        for data, vis, target in zip(batch["image_resized"], batch["image_mask"], batch["target_class"]):
            data = data.unsqueeze(0)
            vis = vis.unsqueeze(0)
            target = target.unsqueeze(0)
            vars_dict = move_to_device_data_vis_and_target(data=data, target=target, vis=vis)
            data, target, vis = vars_dict["data"], vars_dict["target"], vars_dict["vis"]

            if is_convenet:
                norm_data = normalize(data.clone(), mean=CONVENT_NORMALIZATION_MEAN, std=CONVNET_NORMALIZATION_STD)
            else:
                norm_data = normalize(data.clone())
            if verbose:
                plot_image(data)
            pred = model(norm_data)
            pred_logits = pred.logits if type(pred) is ImageClassifierOutput else pred
            pred_probabilities = torch.softmax(pred_logits, dim=1)

            target_probs = torch.gather(pred_probabilities, 1, target[:, None])[:, 0]
            pred_class = pred_probabilities.max(1, keepdim=True)[1].squeeze(1)
            tgt_pred = (target == pred_class).type(target.type()).cpu().numpy()
            num_correct_model[model_index:model_index + len(tgt_pred)] = tgt_pred
            prob_correct_model[model_index:model_index + len(target_probs)] = target_probs.item()

            if verbose:
                print(
                    f'\nOriginal Image. Target: {target.item()}. Top Class: {pred_logits[0].argmax(dim=0).item()}, Max logits: {round(pred_logits[0].max(dim=0)[0].item(), 2)}, Max prob: {round(pred_probabilities[0].max(dim=0)[0].item(), 5)}; Correct class logit: {round(pred_logits[0][target].item(), 2)} Correct class prob: {round(pred_probabilities[0][target].item(), 5)}')

            org_shape = data.shape  # Save original shape

            if perturbation_type == 'NEG':
                vis = -vis
            elif perturbation_type == 'POS':
                vis = vis
            else:
                raise (NotImplementedError(f'perturbation_type config {perturbation_type} not exists'))

            vis = vis.reshape(org_shape[0], -1)

            for perturbation_step in range(len(perturbation_steps)):
                _data = data.clone()
                _data = get_perturbated_data(vis=vis,
                                             image=_data,
                                             perturbation_step=perturbation_steps[perturbation_step],
                                             base_size=base_size,
                                             img_size=img_size)

                if is_convenet:
                    _norm_data = normalize(_data.clone(),
                                           mean=CONVENT_NORMALIZATION_MEAN,
                                           std=CONVNET_NORMALIZATION_STD)
                else:
                    _norm_data = normalize(_data.clone())
                if verbose and perturbation_step < 3:
                    plot_image(_data)

                out = model(_norm_data)
                out_logits = out.logits if type(out) is ImageClassifierOutput else out

                # Target-Class Comparison Accuracy AUC
                target_class_pertub = out_logits.max(1, keepdim=True)[1].squeeze(1)
                temp = (target == target_class_pertub).type(target.type()).cpu().numpy()
                num_correct_pertub[perturbation_step, perturb_index:perturb_index + len(
                    temp)] = temp  # num_correct_pertub is matrix of each row represents perurbation step. Each column represents masked image

                # Target-Class Probability AUC
                perturbation_probabilities = torch.softmax(out_logits, dim=1)
                target_probs = torch.gather(perturbation_probabilities, 1, target[:, None])[:, 0]
                prob_pertub[perturbation_step, perturb_index:perturb_index + len(target_probs)] = target_probs.item()

                if verbose:
                    print(
                        f'{100 * perturbation_steps[perturbation_step]}% pixels blacked. Top Class: {out_logits[0].argmax(dim=0).item()}, Max logits: {round(out_logits[0].max(dim=0)[0].item(), 2)}, Max prob: {round(perturbation_probabilities[0].max(dim=0)[0].item(), 5)}; Correct class logit: {round(out_logits[0][target].item(), 2)} Correct class prob: {round(perturbation_probabilities[0][target].item(), 5)}')

            model_index += len(target)
            perturb_index += len(target)
    auc_perturbation = get_auc(num_correct_pertub=num_correct_pertub, num_correct_model=num_correct_model)
    if is_calculate_deletion_insertion:
        auc_deletion_insertion = get_auc(num_correct_pertub=prob_pertub, num_correct_model=prob_correct_model)
        return auc_perturbation, auc_deletion_insertion
    return auc_perturbation


def get_auc(num_correct_pertub, num_correct_model):
    mean_accuracy_by_step = np.mean(num_correct_pertub, axis=1)
    mean_accuracy_by_step = np.insert(mean_accuracy_by_step, 0, np.mean(num_correct_model))
    auc = calculate_auc(mean_accuracy_by_step=mean_accuracy_by_step) * 100
    return auc


def get_perturbated_data(vis: Tensor, image: Tensor, perturbation_step: Union[float, int], base_size: int,
                         img_size: int):
    """
    vis - Masking of the image (1, 224, 224)
    pic - original image (3, 224, 224)
    """
    _data = image.clone()
    org_shape = (1, 3, img_size, img_size)
    _, idx = torch.topk(vis, int(base_size * perturbation_step), dim=-1)  # vis.shape (50176) / 2 = 25088
    idx = idx.unsqueeze(1).repeat(1, org_shape[1], 1)
    _data = _data.reshape(org_shape[0], org_shape[1], -1)
    _data = _data.scatter_(-1, idx.reshape(1, org_shape[1], -1), 0)
    _data = _data.reshape(*org_shape)
    return _data


def move_to_device_data_vis_and_target(data, target=None, vis=None):
    data = data.to(device)
    vis = vis.to(device)
    if target is not None:
        target = target.to(device)
        return dict(data=data, target=target, vis=vis)
    return dict(data=data, vis=vis)


def update_results_df(results_df: pd.DataFrame, vis_type: str, auc: float):
    return results_df.append({'vis_type': vis_type, 'auc': auc}, ignore_index=True)


import pickle


def save_obj_to_disk(path, obj) -> None:
    if type(path) == str and path[-4:] != '.pkl':
        path += '.pkl'

    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_best_auc_objects_to_disk(path, auc: float, vis, original_image, epoch_idx: int) -> None:
    object = {'auc': auc, 'vis': vis, 'original_image': original_image, 'epoch_idx': epoch_idx}
    save_obj_to_disk(path=path, obj=object)


def run_perturbation_test_opt(model,
                              outputs,
                              stage: str,
                              epoch_idx: int,
                              is_convnet: bool,
                              verbose: bool,
                              img_size: int,
                              experiment_path=None,
                              perturbation_type: str = "POS",
                              ):
    if experiment_path is None:
        experiment_path = Path(EXPERIMENTS_FOLDER_PATH, vit_config['evaluation']['experiment_folder_name'])
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path, exist_ok=True)

    model.eval()
    vit_type_experiment_path = Path(experiment_path, f'{stage}_vis_seg_cls_epoch_{epoch_idx}')
    auc = eval_perturbation_test(experiment_dir=vit_type_experiment_path,
                                 model=model,
                                 outputs=outputs,
                                 img_size=img_size,
                                 is_convenet=is_convnet,
                                 verbose=verbose,
                                 perturbation_type=perturbation_type,
                                 )
    return auc


def plot_image(image) -> None:  # [1,3,224,224] or [3,224,224]
    image = image if len(image.shape) == 3 else image.squeeze(0)
    plt.imshow(image.cpu().detach().permute(1, 2, 0))
    plt.show();


def run_perturbation_test(model,
                          outputs,
                          stage: str,
                          epoch_idx: int,
                          experiment_path,
                          is_convnet: bool,
                          verbose: bool,
                          img_size: int,
                          ):
    VIS_TYPES = [f'{stage}_vis_seg_cls_epoch_{epoch_idx}']

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path, exist_ok=True)
    output_csv_path = Path(experiment_path, f'{stage}_results_df.csv')
    if os.path.exists(output_csv_path):
        results_df = pd.read_csv(output_csv_path)
    else:
        results_df = pd.DataFrame(columns=['vis_type', 'auc'])
    model.to(device)
    model.eval()
    for vis_type in VIS_TYPES:
        print(vis_type)
        vit_type_experiment_path = Path(experiment_path, vis_type)
        auc = eval_perturbation_test(experiment_dir=vit_type_experiment_path,
                                     model=model,
                                     outputs=outputs,
                                     img_size=img_size,
                                     is_convenet=is_convnet,
                                     verbose=verbose,
                                     )
        results_df = update_results_df(results_df=results_df, vis_type=vis_type, auc=auc)
        print(results_df)
        results_df.to_csv(output_csv_path, index=False)
        print(f"Saved results at: {output_csv_path}")
    return auc
