import subprocess
import os
#os.environ['OMP_NUM_THREADS'] = '1'

indices = list(range(0, 16, 2))
indices = indices[:9]
n_processes = 2
total = 11873
indices = list(range(0, total, total//n_processes))
print(indices)

result_path = '../squad_eval.v4.part-{}.jsonl'
processes = []
for i, (start_idx, end_idx) in enumerate(zip(indices, indices[1:])):
    cmd_to_run = [
        './venv/bin/python3.9', 'src/eval.py',
        '--model_path', 'EleutherAI/gpt-neo-1.3B',
        '--state_dict_path', './models/v4/gpt-neo-1.3B-18000.bin',
        '--result_path', result_path.format(i),
        '--start_idx', start_idx,
        '--end_idx', end_idx,
    ]
    cmd_to_run = [str(x) for x in cmd_to_run]

    print('Running:', ' '.join((cmd_to_run)))
    processes.append(subprocess.Popen(cmd_to_run, env=os.environ))

for p in processes: p.wait()
