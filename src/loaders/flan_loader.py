import json
import os
import sys

from tqdm import tqdm

sys.path.append('.')
from src.db_model import load_data

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

"""Dataset downloaded from huggingface datasets from https://huggingface.co/datasets/Muennighoff/flan"""
DATA_DIR = os.environ["FLAN_DATA_DIR"]

for split in ('train', 'test', 'validation'):
    dir = os.path.join(DATA_DIR, split)
    for file in tqdm(os.listdir(dir), desc='Loading datasets from ' + split):
        if file.endswith('.jsonl'):
            dataset_path = os.path.join(dir, file)
            dataset = load_jsonl(dataset_path)
            new_dataset = []
            for data in dataset:
                new_data = dict(
                    input=data['inputs'],
                    target=data['targets'],
                    task=data['task'],
                    dataset='flan',
                )
                new_dataset.append(new_data)
            load_data(new_dataset)
