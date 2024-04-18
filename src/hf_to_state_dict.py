import argparse
import torch

from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument('hf_model_path')
parser.add_argument('save_location')
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.hf_model_path)
torch.save(model.state_dict(), args.save_location)
