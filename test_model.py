import accelerate
import argparse
import sys
import torch
import time

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('base_model', help='Base model config')
parser.add_argument('state_dict_path', help='State dict path')
parser.add_argument('--quantize', action='store_true')
parser.add_argument('--timeit', action='store_true')
args = parser.parse_args()

print('Base Model:', args.base_model)
print('State dict path:', args.state_dict_path)

tokenizer = AutoTokenizer.from_pretrained(args.base_model)
config = AutoConfig.from_pretrained(args.base_model)
with accelerate.init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
print('Model declared')
model = accelerate.load_checkpoint_and_dispatch(
    model,
    checkpoint=args.state_dict_path,
    device_map='auto',
    offload_state_dict=False,
)
print('Weights loaded')
model.eval()

# ======= Quantize ====
if args.quantize:
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print('Model quantized')

# ======= Generate utils ========

def print_wd(text):
    sys.stdout.write(text)
    sys.stdout.flush()

temperature = 0.7
MAX_LEN = 2048
@torch.no_grad()
def my_generate(text:str):
    decoder_input_ids = torch.tensor(tokenizer.encode(text)).view(1, -1)
    past_key_values = None
    for i in range(MAX_LEN - decoder_input_ids.size(1)):
        start = time.monotonic()
        decoder_op = model(
            input_ids = decoder_input_ids,
            use_cache=True,
            past_key_values=past_key_values
        )
        end = time.monotonic()
        if args.timeit: print(f'Time taken for pred: {(end-start)*1e3:0.3f} ms')
        past_key_values = decoder_op.past_key_values
        log_probs = decoder_op.logits[0, -1]
        probs = (log_probs / temperature).softmax(dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        decoder_input_ids = next_token_id.view(1, 1)
        next_token = tokenizer.decode([next_token_id.item()])
        print_wd(next_token)
        text += next_token
        if text.endswith('\nUser:') or next_token_id.item() == tokenizer.eos_token_id:
            break
    return text
        

prefix = '''You are ChatGPT developed by Quantiphi. You are smart, reliable and polite.
You can help users by solving and answering their query. You are trained to follow their instructions.

User: What is your name?
ChatGPT: My name is ChatGPT

User:'''
suffix = '\nChatGPT: '

try:
    print_wd(prefix)
    while True:
        text = input(' ') if prefix.endswith('\nUser:') else input('\nUser: ')
        inp = prefix + text + suffix
        print_wd(suffix[1:])
        prefix = my_generate(inp)
except KeyboardInterrupt as e:
    print('Exiting ...')

