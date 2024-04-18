import accelerate
import argparse
import os
import torch

from transformers import AutoConfig, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument('--model_config')
parser.add_argument('--state_dict_path')
parser.add_argument('--quantized_state_dict_path')
args = parser.parse_args()


config = AutoConfig.from_pretrained(args.model_config)
with accelerate.init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

model = accelerate.load_checkpoint_and_dispatch(
    model,
    args.state_dict_path,
    device_map='auto',
    offload_state_dict=False,
)
model.eval()
print('Model loaded')

# Following is a dynamic quantization. It is the easiest to apply
# The output is W8A32 but model, where W=>weights & A=>Activations
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
print('Model quantized')
torch.save(quantized_model.state_dict(), args.quantized_state_dict_path)
print('Quantized model state_dict saved')
