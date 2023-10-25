from typing import Tuple

import torch
from transformers import ViTForImageClassification
from models.modeling_vit_patch_classification import ViTForMaskGeneration

from vit_loader.ViT_new import vit_base_patch16_224

DEFAULT_MODEL_NAME = 'google/vit-base-patch16-224'


def load_hila_model(model_name: str):
    model = vit_base_patch16_224(pretrained=True).cuda()
    return model


def load_vit_pretrained_for_explaniee(model_name: str) -> ViTForImageClassification:
    hila_model = load_hila_model(model_name=model_name)
    vit_for_image_classification = ViTForImageClassification.from_pretrained(model_name)
    vit_for_image_classification = load_state_of_two_src_models(src_model=hila_model,
                                                                dst_model=vit_for_image_classification)
    del hila_model
    return vit_for_image_classification


def load_vit_pretrained_for_explanier(model_name: str) -> ViTForMaskGeneration:
    hila_model = load_hila_model(model_name=model_name)
    vit_for_image_classification = ViTForImageClassification.from_pretrained(model_name)
    vit_for_patch_classification = ViTForMaskGeneration.from_pretrained(model_name)
    vit_for_image_classification = load_state_of_two_src_models(src_model=hila_model,
                                                                dst_model=vit_for_image_classification)

    vit_for_patch_classification.vit.load_state_dict(vit_for_image_classification.vit.state_dict())
    del hila_model, vit_for_image_classification
    return vit_for_patch_classification


def load_vit_pretrained(model_name: str) -> Tuple[ViTForImageClassification, ViTForMaskGeneration]:
    hila_model = load_hila_model(model_name=model_name)
    vit_for_image_classification = ViTForImageClassification.from_pretrained(model_name)
    vit_for_patch_classification = ViTForMaskGeneration.from_pretrained(model_name)
    vit_for_image_classification = load_state_of_two_src_models(src_model=hila_model,
                                                                dst_model=vit_for_image_classification)

    vit_for_patch_classification.vit.load_state_dict(vit_for_image_classification.vit.state_dict())
    return vit_for_image_classification, vit_for_patch_classification


def calculate_qkv_weights_bias(src_model, layer_idx: int, embed_size: int = 768):
    query_mat = src_model.blocks[layer_idx].attn.qkv.state_dict()['weight'].reshape((3, embed_size, embed_size))[0]
    query_bias = src_model.blocks[layer_idx].attn.qkv.state_dict()['bias'].reshape((3, embed_size))[0]

    key_mat = src_model.blocks[layer_idx].attn.qkv.state_dict()['weight'].reshape((3, embed_size, embed_size))[1]
    key_bias = src_model.blocks[layer_idx].attn.qkv.state_dict()['bias'].reshape((3, embed_size))[1]

    value_mat = src_model.blocks[layer_idx].attn.qkv.state_dict()['weight'].reshape((3, embed_size, embed_size))[2]
    value_bias = src_model.blocks[layer_idx].attn.qkv.state_dict()['bias'].reshape((3, embed_size))[2]

    return query_mat, query_bias, key_mat, key_bias, value_mat, value_bias


def load_state_of_two_src_models(src_model, dst_model):
    dst_model.vit.embeddings.patch_embeddings.projection.load_state_dict(src_model.patch_embed.proj.state_dict())

    for layer_idx in range(12):
        query_mat, query_bias, key_mat, key_bias, value_mat, value_bias = calculate_qkv_weights_bias(src_model,
                                                                                                     layer_idx)

        dst_model.vit.encoder.layer[layer_idx].layernorm_before.load_state_dict(
            src_model.blocks[layer_idx].norm1.state_dict())
        dst_model.vit.encoder.layer[layer_idx].layernorm_after.load_state_dict(
            src_model.blocks[layer_idx].norm2.state_dict())

        dst_model.vit.encoder.layer[layer_idx].attention.attention.query.load_state_dict({'weight': query_mat,
                                                                                          'bias': query_bias})

        dst_model.vit.encoder.layer[layer_idx].attention.attention.key.load_state_dict({'weight': key_mat,
                                                                                        'bias': key_bias})

        dst_model.vit.encoder.layer[layer_idx].attention.attention.value.load_state_dict({'weight': value_mat,
                                                                                          'bias': value_bias})

        dst_model.vit.encoder.layer[layer_idx].attention.output.dense.load_state_dict(
            src_model.blocks[layer_idx].attn.proj.state_dict())

        dst_model.vit.encoder.layer[layer_idx].intermediate.dense.load_state_dict(
            src_model.blocks[layer_idx].mlp.fc1.state_dict())

        dst_model.vit.encoder.layer[layer_idx].output.dense.load_state_dict(
            src_model.blocks[layer_idx].mlp.fc2.state_dict())

    dst_model.vit.layernorm.load_state_dict(src_model.norm.state_dict())
    dst_model.classifier.load_state_dict(src_model.head.state_dict())

    return dst_model
