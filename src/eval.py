import accelerate
import argparse
import json
import os
import onnxruntime as ort
import sys
import time
import torch

from functools import lru_cache
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.append('.')
from src.generation import TextGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--model_config')
parser.add_argument('--weights_file')
parser.add_argument('--result_path')
parser.add_argument('--test_set')
parser.add_argument('--is_onnx', action='store_true')
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.model_config)
if args.is_onnx:
    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 8
    sess_opt.execution_mode  = ort.ExecutionMode.ORT_PARALLEL
    sess_opt.inter_op_num_threads = 8

    model = ort.InferenceSession(args.weights_file, sess_opt)
else:
    config = AutoConfig.from_pretrained(args.model_config)
    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    model = accelerate.load_checkpoint_and_dispatch(
        model,
        args.weights_file,
        device_map='auto',
        offload_state_dict=False,
        #dtype='float16',  setting this to false because on cpu the ops are implemented for fp32 only
    )
    model.eval()
print('Model loaded')
generator = TextGenerator(model=model, tokenizer=tokenizer, is_onnx=args.is_onnx)

def print_wd(text: str):
    sys.stdout.write(text)
    sys.stdout.flush()


with open(args.test_set, 'r') as f:
    squad2 = [json.loads(x) for x in f.read().split('\n') if x != '']

prompt = '''Based on the context provided, extract and return the relevant answer for the question. You are allowed to say "No Comment" if answer is not present in the context.

Context: New Delhi is capital of India.
Question What is the capital of India?
Answer: New Delhi.

Context: {}
Question: {}
Answer:'''

# already inferred ids
if os.path.exists(args.result_path):
    with open(args.result_path, 'r') as f:
        already_inferred_ids = [json.loads(x)['id'] for x in f.read().split('\n')[:-1]] 
else:
    open(args.result_path, 'w').close()
    already_inferred_ids = set()

STOP_TOKENS = ['\nAnswer:', '\nQuestion:', '\nContext:', tokenizer.eos_token]
try:
    output = open(args.result_path, 'a')
    #for i in tqdm(range(args.start_idx, args.end_idx)):
    for i in tqdm(range(len(squad2))):
        ex = squad2[i]
        # check if already inferred
        if ex['id'] in already_inferred_ids: continue 

        inp = prompt.format(ex['context'], ex['question'])
        response = generator.generate(
            inp,
            temperature=0.7,
            max_new_tokens=100,
            stop_tokens=STOP_TOKENS,
        )

        ex['input_prompt'] = inp
        ex['output'] = response
        output.write(json.dumps(ex) + '\n')
except KeyboardInterrupt as e:
    print('exiting ...')
except Exception as e:
    print(e)
finally:
    output.close()

