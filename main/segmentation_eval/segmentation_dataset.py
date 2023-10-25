from typing import Union
from main.segmentation_eval.VOC import VOCSegmentation
from main.segmentation_eval.coco_segmentation import CocoSegmentation
import torch
from pytorch_lightning import seed_everything
from config import config
from main.segmentation_eval.imagenet import ImagenetSegmentation
from utils.consts import IMAGENET_SEG_PATH, COCO_SEG_PATH, VOC_PATH
import gc
from PIL import ImageFile
gc.collect()
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def get_segmentation_dataset(dataset_type: str,
                             batch_size: int,
                             test_img_trans,
                             test_img_trans_only_resize,
                             test_lbl_trans,
                             ) -> Union[CocoSegmentation, VOCSegmentation, ImagenetSegmentation]:
    if dataset_type == 'coco':
        ds = CocoSegmentation(COCO_SEG_PATH,
                              transform=test_img_trans,
                              transform_resize=test_img_trans_only_resize,
                              target_transform=test_lbl_trans)

    elif dataset_type == 'voc':
        ds = VOCSegmentation(root=VOC_PATH,
                             year='2012',
                             image_set='val',
                             download=True,
                             transform=test_img_trans,
                             transform_resize=test_img_trans_only_resize,
                             target_transform=test_lbl_trans)
    elif dataset_type == "imagenet":
        ds = ImagenetSegmentation(path=IMAGENET_SEG_PATH,
                                  batch_size=batch_size,
                                  transform=test_img_trans,
                                  transform_resize=test_img_trans_only_resize,
                                  target_transform=test_lbl_trans,
                                  )
    else:
        raise NotImplementedError(f"No dataset {dataset_type} implemented")
    return ds
