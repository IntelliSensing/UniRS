import os
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

# TODO Add Change Encoder following Chg2Cap Project
def build_change_encoder(
        config: PretrainedConfig):
    if config.chg:
        if config.chg_type == 'Chg2Cap':
            change_encoder = Change_encoder()