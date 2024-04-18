import json
import sys

sys.path.append('.')
from src.db_model import load_data

def load_json(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)


filepath = sys.argv[1]
print('Dataset at:', filepath)
print('Loading into mem ...')
dataset = []
data = load_json(filepath)
for ex in data:
    inp = ex['instruction']
    if ex['input']:
        inp += '\n\n' + ex['input']
    inp += '\n\n'
    dataset.append(dict(
        input = inp,
        target = ex['output'],
        task = '',
        dataset = 'alpaca',
    ))


print('Loading into DB ...')
load_data(dataset)
print('Done')
