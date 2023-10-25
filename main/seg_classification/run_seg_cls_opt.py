import argparse
import os
from distutils.util import strtobool
from main.seg_classification.model_types_loading import load_explainer_explaniee_models_and_feature_extractor, \
    CONVNET_MODELS_BY_NAME
from tqdm import tqdm
from main.seg_classification.image_classification_with_token_classification_model_opt import \
    OptImageClassificationWithTokenClassificationModel
from main.seg_classification.image_token_data_module_opt import ImageSegOptDataModule
from config import config
from main.seg_classification.seg_cls_utils import load_pickles_and_calculate_auc, create_folder_hierarchy, \
    get_gt_classes
from utils import remove_old_results_dfs
from pathlib import Path
import pytorch_lightning as pl
from utils.consts import (
    IMAGENET_VAL_IMAGES_FOLDER_PATH,
    EXPERIMENTS_FOLDER_PATH,
    RESULTS_PICKLES_FOLDER_PATH,
    GT_VALIDATION_PATH_LABELS,
    MODEL_OPTIONS, MODEL_ALIAS_MAPPING,
)
from utils.vit_utils import (
    get_warmup_steps_and_total_training_steps,
    freeze_multitask_model,
    print_number_of_trainable_and_not_trainable_params,
    get_loss_multipliers,
    get_params_from_config,
    suppress_warnings,
    get_backbone_details,
)
from pytorch_lightning import seed_everything
import gc
from PIL import ImageFile
from icecream import ic

suppress_warnings()
seed_everything(config["general"]["seed"])
ImageFile.LOAD_TRUNCATED_IMAGES = True
gc.collect()

if __name__ == '__main__':
    """
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=./:$PYTHONPATH nohup python main/seg_classification/run_seg_cls_opt.py --RUN-BASE-MODEL False --optimize-by-pos False --explainer-model-name resnet --explainee-model-name resnet --train-model-by-target-gt-class True &> nohups_logs/journal/eval/opt_neg/resnet_resnet_stage_b_target_opt_neg.out &
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=./:$PYTHONPATH nohup python main/seg_classification/run_seg_cls_opt.py --RUN-BASE-MODEL False --optimize-by-pos False --explainer-model-name resnet --explainee-model-name resnet --train-model-by-target-gt-class False &> nohups_logs/journal/eval/opt_neg/resnet_resnet_stage_b_predicted_opt_neg.out &
    ll base_model/objects_pkl/ | grep .pkl | wc -l
    """

    params_config = get_params_from_config(config_vit=config["vit"])
    parser = argparse.ArgumentParser(description='Fine-tune EEA model')
    parser.add_argument('--explainer-model-name', type=str, default="resnet", choices=MODEL_OPTIONS)
    parser.add_argument('--explainee-model-name', type=str, default="resnet", choices=MODEL_OPTIONS)
    parser.add_argument("--train-model-by-target-gt-class",
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=params_config["train_model_by_target_gt_class"],
                        )
    parser.add_argument("--RUN-BASE-MODEL",
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=params_config["RUN_BASE_MODEL"],
                        )
    parser.add_argument("--verbose",
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=params_config["verbose"],
                        )
    parser.add_argument("--optimize-by-pos",
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=params_config["optimize_by_pos"],
                        )

    parser.add_argument('--n_epochs_to_optimize_stage_b',
                        type=int,
                        default=params_config["n_epochs_to_optimize_stage_b"])
    parser.add_argument('--n-epochs', type=int, default=params_config["n_epochs"])
    parser.add_argument('--prediction-loss-mul', type=int, default=params_config["prediction_loss_mul"])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--is-freezing-explaniee-model',
                        type=lambda x: bool(strtobool(x)),
                        nargs="?",
                        const=True,
                        default=params_config["is_freezing_explaniee_model"])
    parser.add_argument('--explainer-model-n-first-layers-to-freeze',
                        type=int,
                        default=params_config["explainer_model_n_first_layers_to_freeze"])
    parser.add_argument('--is-clamp-between-0-to-1',
                        type=lambda x: bool(strtobool(x)),
                        nargs="?",
                        const=True,
                        default=params_config["is_clamp_between_0_to_1"])
    parser.add_argument('--is-competitive-method-transforms',
                        type=lambda x: bool(strtobool(x)),
                        nargs="?",
                        const=True,
                        default=params_config["is_competitive_method_transforms"])
    parser.add_argument('--plot-path', type=str, default=params_config["plot_path"])
    parser.add_argument('--default-root-dir', type=str, default=params_config["default_root_dir"])
    parser.add_argument('--mask-loss', type=str, default=params_config["mask_loss"])
    parser.add_argument('--train-n-label-sample', type=str, default=params_config["train_n_label_sample"])
    parser.add_argument('--lr', type=float, default=params_config["lr"])
    parser.add_argument('--start-epoch-to-evaluate', type=int, default=params_config["start_epoch_to_evaluate"])
    parser.add_argument('--n-batches-to-visualize', type=int, default=params_config["n_batches_to_visualize"])
    parser.add_argument('--is-ce-neg', type=str, default=params_config["is_ce_neg"])
    parser.add_argument('--activation-function', type=str, default=params_config["activation_function"])
    parser.add_argument('--use-logits-only',
                        type=lambda x: bool(strtobool(x)),
                        nargs="?",
                        const=True,
                        default=params_config["use_logits_only"])
    parser.add_argument('--evaluation-experiment-folder-name',
                        type=str,
                        default=params_config["evaluation_experiment_folder_name"])

    args = parser.parse_args()

    EXPLAINEE_MODEL_NAME, EXPLAINER_MODEL_NAME = MODEL_ALIAS_MAPPING[args.explainee_model_name], \
                                                 MODEL_ALIAS_MAPPING[args.explainer_model_name]

    IS_EXPLANIEE_CONVNET = True if EXPLAINEE_MODEL_NAME in CONVNET_MODELS_BY_NAME.keys() else False
    IS_EXPLAINER_CONVNET = True if EXPLAINER_MODEL_NAME in CONVNET_MODELS_BY_NAME.keys() else False

    target_or_predicted_model = "target" if args.train_model_by_target_gt_class else "predicted"

    CKPT_PATH, IMG_SIZE, PATCH_SIZE, MASK_LOSS_MUL, CHECKPOINT_EPOCH_IDX, BASE_CKPT_MODEL_AUC = get_backbone_details(
        explainer_model_name=args.explainer_model_name,
        explainee_model_name=args.explainee_model_name,
        target_or_predicted_model=target_or_predicted_model,
    )

    loss_multipliers = get_loss_multipliers(normalize=False,
                                            mask_loss_mul=MASK_LOSS_MUL,
                                            prediction_loss_mul=args.prediction_loss_mul,
                                            )

    exp_name = f'direct_opt_ckpt_{CHECKPOINT_EPOCH_IDX}_auc_{BASE_CKPT_MODEL_AUC}_explanier_{args.explainer_model_name.replace("/", "_")}__explaniee_{args.explainee_model_name.replace("/", "_")}__opt_by_pos_{args.optimize_by_pos}__pred_{args.prediction_loss_mul}_mask_l_{args.mask_loss}_{MASK_LOSS_MUL}__train_n_samples_{args.train_n_label_sample * 1000}_lr_{args.lr}_by_target_gt__{args.train_model_by_target_gt_class}'
    plot_path = Path(args.plot_path, exp_name)
    BASE_AUC_OBJECTS_PATH = Path(RESULTS_PICKLES_FOLDER_PATH,
                                 'target' if args.train_model_by_target_gt_class else 'predicted')

    EXP_PATH = Path(BASE_AUC_OBJECTS_PATH, exp_name)
    ic(EXP_PATH)
    ic(args.explainer_model_name)
    ic(args.optimize_by_pos)
    ic(args.train_model_by_target_gt_class)
    ic(args.n_epochs_to_optimize_stage_b)
    ic(args.RUN_BASE_MODEL)
    ic(MASK_LOSS_MUL)
    ic(args.verbose)
    os.makedirs(EXP_PATH, exist_ok=True)
    BEST_AUC_PLOT_PATH, BEST_AUC_OBJECTS_PATH, BASE_MODEL_BEST_AUC_PLOT_PATH, BASE_MODEL_BEST_AUC_OBJECTS_PATH = create_folder_hierarchy(
        base_auc_objects_path=BASE_AUC_OBJECTS_PATH,
        exp_name=exp_name,
    )

    model_for_classification_image, model_for_mask_generation, feature_extractor = load_explainer_explaniee_models_and_feature_extractor(
        explainee_model_name=EXPLAINEE_MODEL_NAME,
        explainer_model_name=EXPLAINER_MODEL_NAME,
        activation_function=args.activation_function,
        img_size=IMG_SIZE,
    )

    warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
        n_epochs=args.n_epochs_to_optimize_stage_b,
        train_samples_length=len(list(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH).iterdir())),
        batch_size=args.batch_size,
    )

    model = OptImageClassificationWithTokenClassificationModel(
        model_for_classification_image=model_for_classification_image,
        model_for_mask_generation=model_for_mask_generation,
        is_clamp_between_0_to_1=args.is_clamp_between_0_to_1,
        plot_path=plot_path,
        warmup_steps=warmup_steps,
        total_training_steps=total_training_steps,
        best_auc_objects_path=BASE_MODEL_BEST_AUC_OBJECTS_PATH if args.RUN_BASE_MODEL else BEST_AUC_OBJECTS_PATH,
        checkpoint_epoch_idx=CHECKPOINT_EPOCH_IDX,
        best_auc_plot_path=BASE_MODEL_BEST_AUC_PLOT_PATH if args.RUN_BASE_MODEL else BEST_AUC_PLOT_PATH,
        run_base_model_only=args.RUN_BASE_MODEL,
        is_explainer_convnet=IS_EXPLAINER_CONVNET,
        is_explainee_convnet=IS_EXPLANIEE_CONVNET,
        lr=args.lr,
        n_epochs=CHECKPOINT_EPOCH_IDX + args.n_epochs_to_optimize_stage_b,
        start_epoch_to_evaluate=args.start_epoch_to_evaluate,
        train_model_by_target_gt_class=args.train_model_by_target_gt_class,
        use_logits_only=args.use_logits_only,
        n_batches_to_visualize=args.n_batches_to_visualize,
        mask_loss=args.mask_loss,
        mask_loss_mul=MASK_LOSS_MUL,
        prediction_loss_mul=args.prediction_loss_mul,
        activation_function=args.activation_function,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        is_ce_neg=args.is_ce_neg,
        verbose=args.verbose,
        optimize_by_pos=args.optimize_by_pos,
    )

    experiment_path = Path(EXPERIMENTS_FOLDER_PATH, args.evaluation_experiment_folder_name)
    remove_old_results_dfs(experiment_path=experiment_path)
    model = freeze_multitask_model(
        model=model,
        is_freezing_explaniee_model=args.is_freezing_explaniee_model,
        explainer_model_n_first_layers_to_freeze=args.explainer_model_n_first_layers_to_freeze,
        is_explainer_convnet=IS_EXPLAINER_CONVNET,
    )
    print(exp_name)
    print_number_of_trainable_and_not_trainable_params(model)

    IMAGES_PATH = IMAGENET_VAL_IMAGES_FOLDER_PATH
    ic(exp_name)
    print(f"Total Images in path: {len(os.listdir(IMAGES_PATH))}")
    ic(MASK_LOSS_MUL, args.prediction_loss_mul)
    images_listdir = sorted(list(Path(IMAGES_PATH).iterdir()))
    targets = get_gt_classes(path=GT_VALIDATION_PATH_LABELS)
    for idx, (image_path, target) in tqdm(enumerate(zip(images_listdir, targets)), position=0, leave=True,
                                          total=len(images_listdir)):
        data_module = ImageSegOptDataModule(
            batch_size=1,
            train_image_path=str(image_path),
            val_image_path=str(image_path),
            target=target,
            is_competitive_method_transforms=args.is_competitive_method_transforms,
            feature_extractor=feature_extractor,
            is_explaniee_convnet=IS_EXPLANIEE_CONVNET,
        )
        trainer = pl.Trainer(
            logger=[],
            accelerator='gpu',
            gpus=1,
            devices=[1, 2],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=100,
            max_epochs=CHECKPOINT_EPOCH_IDX + args.n_epochs_to_optimize_stage_b,
            resume_from_checkpoint=CKPT_PATH,
            enable_progress_bar=False,
            enable_checkpointing=False,
            default_root_dir=args.default_root_dir,
            weights_summary=None
        )
        trainer.fit(model=model, datamodule=data_module)
    mean_auc = load_pickles_and_calculate_auc(
        path=BASE_MODEL_BEST_AUC_OBJECTS_PATH if args.RUN_BASE_MODEL else BEST_AUC_OBJECTS_PATH)
    print(f"Mean AUC: {mean_auc}")
    ic(exp_name)
