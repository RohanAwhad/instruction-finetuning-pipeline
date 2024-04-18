import argparse
import os
import sys
import torch

from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.append('.')
from src.lightning_ml_module import LightningLanguageModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='distilgpt2')
parser.add_argument('--save_dir', type=str)
parser.add_argument('--ckpt_dir_name', type=str)
parser.add_argument('--model_name', type=str, default='pytorch_model.bin')
args = parser.parse_args()

config = AutoConfig.from_pretrained(args.model)
base_model = AutoModelForCausalLM.from_config(config)  # doesn't download weights, saving us mem

model_config = dict(
    model=base_model,
    step_size=None,
    n_warmup=None,
    n_total_iter=None,
    lr=None,
    update_every_n_steps=None,
    strategy=None
)

checkpoint_path = os.path.join(args.save_dir, args.ckpt_dir_name)
single_ckpt_path = os.path.join(args.save_dir, 'deepspeed_to_lightning.bin')
convert_zero_checkpoint_to_fp32_state_dict(checkpoint_path, single_ckpt_path)

loaded_model = LightningLanguageModel.load_from_checkpoint(
    single_ckpt_path,
    strict=False,
    # setting the above arg to false, because there is no `lm_head.weight`
    # key in state_dict (might be because it is shared.)
    # Don't know for sure. solved the error based on
    # this issue: https://github.com/Lightning-AI/lightning/issues/10964
    **model_config
)
torch.save(loaded_model.model.state_dict(), os.path.join(args.save_dir, args.model_name))

print('Done')
