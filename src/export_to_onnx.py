# Testing so this will be minimal
# TODO (rohan): add tests between torch and onnx model outputs
# TODO (rohan): increase the sample text len close to max context size of the model (i.e. 2048)
import accelerate
import argparse
import onnx
import onnxruntime.quantization as quantization
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

parser = argparse.ArgumentParser()
parser.add_argument('--model_config', type=str)
parser.add_argument('--trained_model_path', type=str)
parser.add_argument('--model_path_fp32', type=str)
parser.add_argument('--model_path_quantized', type=str)
parser.add_argument('--max_len', type=int)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_config)
config = AutoConfig.from_pretrained(args.model_config)
if args.trained_model_path:
    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model = accelerate.load_checkpoint_and_dispatch(
        model,
        checkpoint=args.trained_model_path,
        device_map='auto',
        offload_state_dict=False,
    )
else:
    model = AutoModelForCausalLM.from_config(config)
model.eval()

text = '''Quantization is not a loss-less transformation. It may negatively affect a model’s accuracy. A solution to this problem is to compare the weights and activations tensors of the original computation graph vs those of the quantized one, identify where they differ most, and avoid quantizing these tensors, or choose another quantization/calibration method. This is called quantization debugging. To facilitate this process, we provide Python APIs for matching weights and activation tensors between a float32 model and its quantized counterpart.

API for debugging is in module onnxruntime.quantization.qdq_loss_debug, which has the following functions:

Function create_weight_matching(). It takes a float32 model and its quantized model, and output a dictionary that matches the corresponding weights between these two models.
Function modify_model_output_intermediate_tensors(). It takes a float32 or quantized model, and augment it to save all its activations.
Function collect_activations(). It takes a model augmented by modify_model_output_intermediate_tensors(), and an input data reader, runs the augmented model to collect all the activations.
Function create_activation_matching(). You can imagine that you run collect_activations(modify_model_output_intermediate_tensors()) on both the float32 and its quantized model, to collect two sets of activations. This function takes these two set of activations, and matches up corresponding ones, so that they can be easily compared by the user.
In summary, ONNX Runtimes provides Python APIs for matching up corresponding weights and activation tensors between a float32 model and its quantized counterpart. This allows the user to easily compare them to locate where are the biggest differences.

Model optimization during quantization creates difficulties for this debugging process though, since it may changes the computation graph in a significant way, resulting in a quantized model that is drastically different from the original. This makes it hard to match up corresponding tensors from the two models. As a result, we recommend performing model optimization during pre-processing instead of the quantization process.'''


# We do not need a model without past_key_values
# just pass attns and values as np.empty arrays
class ONNXDecoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attns, values):
        past_key_values = tuple(zip(attns, values))
        output = self.model(input_ids, past_key_values=past_key_values)
        logits = output.logits
        past_key_values = output.past_key_values
        attns, values = [], []
        for block in past_key_values:
            attns.append(block[0])
            values.append(block[1])
        
        attns = torch.stack(attns)
        values = torch.stack(values)
        return logits, attns, values



print('Generating example input ...')
input_ids = torch.tensor(tokenizer.encode(text)[-args.max_len:], dtype=torch.long).view(1, -1)
decoder_op = model(input_ids)
logits = decoder_op.logits
past_key_values = decoder_op.past_key_values
attns, values = [], []
for block in past_key_values:
    attns.append(block[0])
    values.append(block[1])

attns = torch.stack(attns)
values = torch.stack(values)
new_input_ids = logits[:, -1, :].argmax(-1, keepdims=True)

onnx_decoder = ONNXDecoder(model)
onnx_decoder.eval()
for param in onnx_decoder.parameters(): param.requires_grad = False
print('Exporting ...')
torch.onnx.export(
    onnx_decoder,
    args=(new_input_ids.long(), attns, values),
    f=args.model_path_fp32,
    input_names=['input_ids', 'input_attns', 'input_values'],
    output_names=['logits', 'attns', 'values'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'input_seq_len'},
        'input_attns': {1: 'batch_size', 3: 'input_pkv_seq_len'},
        'input_values': {1: 'batch_size', 3: 'input_pkv_seq_len'},
        'logits': {0: 'batch_size', 1: 'output_seq_len'},
        'attns': {1: 'batch_size', 3: 'pkv_seq_len'},
        'values': {1: 'batch_size', 3: 'pkv_seq_len'},
    },
    do_constant_folding=True,
    verbose=False,
    opset_version=16,
)
onnx_model = onnx.load(args.model_path_fp32)
#onnx.checker.check_model(onnx_model)

# I am unable to export the quantized model to onnx
# there is an intermediate step where the model is serialized using torchscript
# and the exporting fails there, because some arguments in the model need to be none type
print('='*5, 'Quantization', '='*5)
quantization.quantize_dynamic(
    args.model_path_fp32,
    args.model_path_quantized,
    use_external_data_format=True
)
