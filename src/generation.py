import torch
import numpy as np

class TextGenerator:
    def __init__(self, model, tokenizer, is_onnx: bool=True):
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token = self.tokenizer.eos_token

        self.generate = self._generate_using_onnx if is_onnx else self._generate_using_torch
        if is_onnx:
            input_attn_shape = self.model.get_inputs()[1].shape
            self.num_layers = input_attn_shape[0]
            self.num_heads = input_attn_shape[2]
            self.embd_dim = input_attn_shape[4]
            self.token_ids = np.arange(self.tokenizer.vocab_size)
            self.generate = self._generate_using_onnx
        else:
            self.generate = self._generate_using_torch

    def _generate_using_onnx(
        self,
        text: str,
        past_key_values=None,
        max_new_tokens=100,
        temperature=0.7,
        stop_tokens=None,
        do_stream=False,
    ):
        # Input Validation
        if stop_tokens is None:
            stop_tokens = [self.eos_token]
        elif isinstance(stop_tokens, str):
            stop_tokens = [stop_tokens, self.eos_token]
        elif isinstance(stop_tokens, list) and self.eos_token not in stop_tokens:
            stop_tokens.append(self.eos_token)
        elif not isinstance(stop_tokens, list):
            raise ValueError(f'stop_tokens need to be either a list[str] or str. Found {type(stop_tokens)}')

        # Generation
        ret = ''
        input_ids = np.array(self.tokenizer.encode(text)[-self.tokenizer.model_max_length:], dtype=np.int64).reshape(1, -1)
        attns = np.empty((self.num_layers, 1, self.num_heads, 0, self.embd_dim), dtype=np.float32)
        values = np.empty((self.num_layers, 1, self.num_heads, 0, self.embd_dim), dtype=np.float32)
        break_flag = False
        for i in range(max_new_tokens):
            #start = time.monotonic()
            decoder_op = self.model.run(
                None,
                input_feed = dict(
                    input_ids=input_ids,
                    input_attns = attns,
                    input_values = values,
                )
            )
            #end = time.monotonic()
            #print(f'Time taken: {((end-start) * 1000):0.2f}ms')
            logits, attns, values = decoder_op

            # next token gen
            _tmp = np.exp(logits[0, -1] / temperature)
            probs = _tmp / np.sum(_tmp)
            next_token_id = np.random.choice(self.token_ids, 1, p=probs)
            input_ids = next_token_id.reshape(1, 1)
            token = self.tokenizer.decode([next_token_id[0]])
            if do_stream: yield token
            ret += token
            for st in stop_tokens:
                if ret.endswith(st):
                    ret = ret[:-len(st)]
                    break_flag = True
                    break

            if break_flag:
                break
            #print_wd(token)

        if not do_stream:
            return ret

    @torch.no_grad()
    def _generate_using_torch(
        self,
        text: str,
        past_key_values=None,
        max_new_tokens=100,
        temperature=0.7,
        stop_tokens=None,
        do_stream=False,
    ):
        # Input Validation
        if stop_tokens is None:
            stop_tokens = [self.eos_token]
        elif isinstance(stop_tokens, str):
            stop_tokens = [stop_tokens, self.eos_token]
        elif isinstance(stop_tokens, list) and self.eos_token not in stop_tokens:
            stop_tokens.append(self.eos_token)
        elif not isinstance(stop_tokens, list):
            raise ValueError(f'stop_tokens need to be either a list[str] or str. Found {type(stop_tokens)}')

        # Generation
        ret = ''
        input_ids = torch.tensor(self.tokenizer.encode(text)).view(1, -1)
        break_flag = False
        for i in range(max_new_tokens):
            #start = time.monotonic()
            decoder_op = self.model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=past_key_values
            )
            #end = time.monotonic()
            #print(f'Time taken: {((end-start) * 1000):0.2f}ms')
            past_key_values = decoder_op.past_key_values

            # next token gen
            log_probs = decoder_op.logits[0, -1]
            probs = (log_probs / temperature).softmax(dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            input_ids = next_token_id.view(1, 1)
            token = self.tokenizer.decode([next_token_id.item()])
            if do_stream: yield token
            ret += token
            for st in stop_tokens:
                if ret.endswith(st):
                    ret = ret[:-len(st)]
                    break_flag = True
                    break

            if break_flag:
                break
            #print_wd(token)

        if not do_stream:
            return ret
