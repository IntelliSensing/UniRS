# This file is modified from https://github.com/haotian-liu/LLaVA/

import os

import deepspeed
import torch
from safetensors import safe_open

from .base_projector import MultimodalProjectorConfig, MultimodalProjector
from transformers import PretrainedConfig, PreTrainedModel
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

def get_state_of_change_caption_projector(named_params):
    to_return = {k.replace('layers.', 'layers.single.') if "layers." in k else k: t for k, t in named_params.items()}
    return to_return

def get_state_dict_of_projector(model_path, state_dict):
    with safe_open(model_path, framework="pt", device='cpu') as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    return state_dict

def build_mm_projector(
    model_type_or_path: str, config: PretrainedConfig
) -> PreTrainedModel:
    if model_type_or_path is None:
        return None

    ## load from pretrained model
    # TODO: 修改参数导入方法
    if config.resume_path:
        assert os.path.exists(
            model_type_or_path
        ), f"Resume mm projector path {model_type_or_path} does not exist!"

        mm_projector_cfg = MultimodalProjectorConfig(model_type_or_path)
        mm_projector = MultimodalProjector(mm_projector_cfg, config).to(
            eval(config.model_dtype)
        )
        # print(config.model_dtype)
        # mm_projector = mm_projector.from_pretrained(
        #     model_type_or_path, config, torch_dtype=eval(config.model_dtype), ignore_mismatched_sizes=True
        # )
        model_path = model_type_or_path + '/model.safetensors'
        state_dict = {}
        mm_projector.load_state_dict(get_state_dict_of_projector(model_path, state_dict), strict=False)
        # mm_projector = load_state_dict_from_zero_checkpoint(mm_projector, model_path, tag='')
        return mm_projector

    ## build from scratch
    else:
        mm_projector_cfg = MultimodalProjectorConfig(model_type_or_path)
        mm_projector = MultimodalProjector(mm_projector_cfg, config).to(
            eval(config.model_dtype)
        )
        return mm_projector
