import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

class SFTDataset(Dataset):
    def __init__(self, ids: list[int], tokenizer: PreTrainedTokenizer, dataset, max_len: int = 512, find_batch_size: bool=False):
        super().__init__()
        self.dataset = dataset
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.find_batch_size = find_batch_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        data = self.dataset[self.ids[idx]]
        prompt, target = data.input, data.target

        prompt_ids = self.tokenizer.encode(prompt)
        target_ids = self.tokenizer.encode(target)

        labels = [-100] * (len(prompt)-1)
        labels += target_ids
        labels += [self.tokenizer.eos_token_id]

        inp_ids = prompt_ids + target_ids
        attn_mask = [1] * len(inp_ids)

        # pad to max len to stablize mem usage while training
        # This should be done when figuring out what batch size can hold the
        # worst case scenario
        # in general while training, having variable seq len will
        # increase training efficiency by minimizing unnecessary FLOP
        pad_len = self.max_len - len(inp_ids)
        if self.find_batch_size and pad_len > 0:
            inp_ids += [self.tokenizer.eos_token_id] * pad_len
            attn_mask += [0] * pad_len
            labels += [-100] * pad_len

        assert len(inp_ids) == len(attn_mask) == len(labels)

        inp_ids = inp_ids[-self.max_len:]
        attn_mask = attn_mask[-self.max_len:]
        labels = labels[-self.max_len:]

        return {
            "input_ids": torch.tensor(inp_ids, dtype=torch.int32),
            "attenion_mask": torch.tensor(attn_mask, dtype=torch.int32),
            "labels": torch.tensor(labels, dtype=torch.int32),
        }
