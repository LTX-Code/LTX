import json
import pandas as pd
from PIL import Image
import os
from pathlib import Path
from typing import Any, Dict, List
from torch import Tensor
from main.seg_classification.cnns.cnn_utils import convnet_preprocess
from utils.consts import PICKLES_FOLDER_PATH
from matplotlib import pyplot as plt


def create_df_of_img_name_with_label(path: Path) -> pd.DataFrame:
    dirlist = os.listdir(path)
    dataframes = []
    for dir in dirlist:
        print('dir', dir)
        if os.path.isdir(Path(path, dir)):
            images_path = Path(path, dir, 'images')
            files_names = os.listdir(images_path)
            df = pd.DataFrame(files_names, columns=['file_name'])
            df['label'] = dir
            dataframes.append(df)
    return pd.concat(dataframes)


def _get_class2_idx(path: Path) -> Dict:
    with open(path, 'r') as f:
        class2idx = json.load(f)
    return class2idx


def read_csv(base_path: Path, file_name: str) -> pd.DataFrame:
    class2idx = _get_class2_idx(path=Path(base_path, 'class2idx.json'))
    df = pd.read_csv(Path(base_path, f'{file_name}.csv'), index_col=0)
    df['label'] = df.label.map(lambda x: class2idx[x])
    return df


def read_gt_labels(path: Path) -> List[str]:
    with open(path, 'r') as f:
        return f.readlines()


def parse_gt_labels(labels: List[str]) -> List[int]:
    return [int(label.replace('\n', '')) for label in labels]


def get_image_from_path(path: str) -> Image:
    return Image.open(path)


import pickle


def save_obj_to_disk(path, obj) -> None:
    if type(path) == str and path[-4:] != '.pkl':
        path += '.pkl'

    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(obj_name: str) -> Any:
    with open(Path(f"{PICKLES_FOLDER_PATH}", f"{obj_name}.pkl"), 'rb') as f:
        return pickle.load(f)


def remove_old_results_dfs(experiment_path: str):
    stages = ['train', 'val']
    for stage in stages:
        output_csv_path = Path(experiment_path, f'{stage}_results_df.csv')
        if os.path.exists(output_csv_path):
            os.remove(output_csv_path)


def get_gt_classes(path):
    with open(path, 'r') as f:
        gt_classes_list = f.readlines()
    gt_classes_list = [int(record.split()[-1].replace('\n', '')) for record in gt_classes_list]
    return gt_classes_list


def show_image(image, title: str = None):
    plt.imshow(image)
    plt.axis('off');
    if title is not None:
        plt.title(title)
    plt.show()


def get_preprocessed_image(image_path: str) -> Tensor:
    image = get_image_from_path(path=image_path)
    image = image if image.mode == "RGB" else image.convert("RGB")  # Black & White images
    inputs = convnet_preprocess(image).unsqueeze(0)
    return inputs
