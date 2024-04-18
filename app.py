import accelerate
import argparse
import os
import onnxruntime as ort
import uvicorn

from fastapi import FastAPI, status
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.generation import TextGenerator

app = FastAPI()
port = 5000

# Model Initalization
model_path = 'EleutherAI/gpt-neo-1.3B'
#model_path = '/Users/rohan/3 Resources/testing_models/tiny-random-gpt_neo/'
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(' ======= Model Loading =========')
is_onnx = True
#weights_path = None
#weights_path = 'models/v5/gpt-neo-1.3B-3epochs.bin'
weights_path = 'models/v5/gpt-neo-1.3B-3epochs-quantized/model.onnx'
if is_onnx:
    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 8
    sess_opt.execution_mode  = ort.ExecutionMode.ORT_PARALLEL
    sess_opt.inter_op_num_threads = 8

    model = ort.InferenceSession(weights_path, sess_opt)

elif weights_path:
    config = AutoConfig.from_pretrained(model_path)
    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    print('Model declared')
    model = accelerate.load_checkpoint_and_dispatch(
        model,
        checkpoint=weights_path,
        device_map='auto',
        offload_state_dict=False,
    )
    model.eval()

else:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

print('Weights loaded')
# ==============

generator = TextGenerator(model, tokenizer, is_onnx)
# =======

class RequestModel(BaseModel):
    prompt: str
    stop_tokens: list[str]
    temperature: float=0.7
    max_new_tokens: int=100
    do_stream: bool=True

@app.post('/generate')
def generate_post(inp: RequestModel):
    contents = generator.generate(
        text=inp.prompt,
        temperature=inp.temperature,
        stop_tokens=inp.stop_tokens,
        max_new_tokens=inp.max_new_tokens,
        do_stream=inp.do_stream,
    )
    if inp.do_stream:
        response = StreamingResponse(
            content=contents,
            status_code=status.HTTP_200_OK,
            media_type='text/html'
        )
        return response
    else:
        return contents


@app.get('/generate')
def generate_path(prompt: str):
    return generator.generate(prompt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--port', type=int)
    args = parser.parse_args()

    port = args.port if args.port else port
    if args.reload:
        uvicorn.run('app:app', host='localhost', port=port, reload=True)
    else:
        uvicorn.run(app, host='0.0.0.0', port=port)
