import argparse
import evaluate
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('predictions')
args = parser.parse_args()

if args.predictions.endswith('jsonl'):
    with open(args.predictions, 'r') as f:
        preds = [json.loads(x) for x in f.read().split('\n')[:-1]]

    for x in preds:
        if isinstance(x['answers'], dict): x['answers'] = ' | '.join(x['answers']['text'])
        if x['answers'] == '': x['answers'] = 'Not in Background | No Comment'

    df = pd.DataFrame(preds)
    df.to_csv('.'.join(args.predictions.split('.')[:-1] + ['csv', ]))
else:
    df = pd.read_csv(args.predictions, index_col=0)

rouge = evaluate.load('rouge')
print('Rouge Scores:', rouge.compute(
    predictions=df['output'].str.strip().tolist(),
    references=df['answers'].str.strip().tolist(),
))

