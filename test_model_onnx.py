import argparse
import onnx
import onnxruntime as ort
import numpy as np
import sys
import torch
import time

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('tokenizer', help='Tokenizer config')
parser.add_argument('onnx_model', help='ONNX model path')
args = parser.parse_args()

print('Tokenizer:', args.tokenizer)
print('ONNX Model path:', args.onnx_model)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
model = ort.InferenceSession(args.onnx_model)
print('Inference Session loaded')
attn_shape = model.get_inputs()[1].shape
NUM_BLOCKS = attn_shape[0]
NUM_HEADS = attn_shape[2]
EMBD_DIM = attn_shape[4]

# ======= Generate utils ========

def print_wd(text):
    sys.stdout.write(text)
    sys.stdout.flush()

temperature = 0.7
MAX_LEN = 2048
MAX_NEW_TOKENS = 100
TOKEN_IDS = np.arange(tokenizer.vocab_size)
@torch.no_grad()
def my_generate(text:str):
    input_ids = np.array(tokenizer.encode(text)[-MAX_LEN:], dtype=np.int64).reshape(1, -1)
    attns = np.empty((NUM_BLOCKS, 1, NUM_HEADS, 0, EMBD_DIM), dtype=np.float32)
    values = np.empty((NUM_BLOCKS, 1, NUM_HEADS, 0, EMBD_DIM), dtype=np.float32)
    for i in range(MAX_NEW_TOKENS):
        start = time.monotonic()
        decoder_op = model.run(
            None,
            input_feed = dict(
                input_ids = input_ids,
                input_attns = attns,
                input_values = values,
            )
        )
        end = time.monotonic()
        print(f'Time Taken: {(end-start)*1e3:0.2f} ms')
        logits, attns, values = decoder_op
        log_probs = logits[0, -1] / temperature
        log_probs_exp = np.exp(log_probs)
        probs = log_probs_exp / np.sum(log_probs_exp)
        next_token_id = np.random.choice(TOKEN_IDS, 1, p=probs)
        input_ids = next_token_id.reshape(1, 1)
        next_token = tokenizer.decode([next_token_id[0]])
        print_wd(next_token)
        text += next_token
        if text.endswith('\nUser:') or next_token_id[0] == tokenizer.eos_token_id:
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

