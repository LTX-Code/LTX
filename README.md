# PyTorch Implementation of Learning to Explain: A Model-Agnostic Framework for Explaining Black Box Models

We present Learning to Explain (LTX), a model-agnostic framework designed for providing post-hoc explanations for vision models. The LTX framework introduces an ``explainer'' model that generates explanation maps, highlighting the crucial regions that justify the predictions made by the model being explained. To train the explainer, we employ a two-stage process consisting of initial pretraining followed by per-instance finetuning. During both stages of training, we utilize a unique configuration where we compare the explained model's prediction for a masked input with its original prediction for the unmasked input. This approach enables the use of a novel counterfactual objective, which aims to anticipate the model's output using masked versions of the input image. Importantly, the LTX framework is not restricted to a specific model architecture and can provide explanations for both Transformer-based and convolutional models. Through our evaluations, we demonstrate that LTX significantly outperforms the current state-of-the-art in explainability across various metrics.

<img src="images\2_classes_vis_github.png" alt="2_classes_vis_github" width="250" height="200" align:center/>

<img src="images\single_object_vis_github.png" alt="single_object_vis_github" width="200" height="350" align:center />


## Reproducing results on ViT - Perturbations Metrics
---
### Loading Checkpoints:
- Download `checkpoints.zip` from https://drive.google.com/file/d/1syOvmnXFgMsIgu-10LNhm0pHDs2oo1gm/
- unzip classifier.zip -d ./checkpoints/ (after unzipping, the checkpoints should be in the corresponding folders based on the backbone's type (`vit_base`))

These checkpoints are essential for reproducing the results. All explanation metrics can be calculated using the mask files created during the LTX procedure.

### Evaluations

#### LTX

```python
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH nohup python main/seg_classification/run_seg_cls_opt.py --RUN-BASE-MODEL False --explainer-model-name vit_base_224 --explainee-model-name vit_base_224 --train-model-by-target-gt-class True
```

#### pLTX

```python
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH nohup python main/seg_classification/run_seg_cls_opt.py --RUN-BASE-MODEL True --explainer-model-name vit_base_224 --explainee-model-name vit_base_224 --train-model-by-target-gt-class True
```
### Pretraining Phase - pLTX model

```python
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH nohup python main/seg_classification/run_seg_cls.py --enable-checkpointing True --explainer-model-name vit_base_224 --explainee-model-name vit_base_224 --mask-loss-mul 50 --train-model-by-target-gt-class True --n-epochs 30 --train-n-label-sample 1
```



## Reproducing results on ViT-Base & ViT-Small - Segmentation Results

---
### Download the segmentation datasets:
- Download imagenet_dataset [Link to download dataset](http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat)
- Download the COCO_Val2017 [Link to download dataset](https://cocodataset.org/#download)
- Download Pascal_val_2012 [Link to download dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

- Move all datasets to ./data/

### pLTX

```python
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH nohup python main/segmentation_eval/seg_stage_a.py --explainer-model-name vit_base_224 --explainee-model-name vit_base_224 --dataset-type imagenet
```

### LTX

```python
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH nohup python main/segmentation_eval/seg_stage_b.py --explainer-model-name vit_base_224 --explainee-model-name vit_base_224 --dataset-type imagenet
```

** The dataset can be chosen by the parameter of `--dataset-type` from `imagenet`, `coco`, `voc`
