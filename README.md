# Instuction Fine-tuning pipeline


Using the Flan dataset available on HF Datasets hub
Check it out [here](https://huggingface.co/datasets/Muennighoff/flan)

Flan dataset is not human like, and also the train loss keeps on diverging
Trying out self-instruct dataset. You can check it out [here](https://github.com/yizhongw/self-instruct)
Dataset used can be downloaded from [here](https://raw.githubusercontent.com/yizhongw/self-instruct/main/data/finetuning/self_instruct_221203/gpt3_finetuning_data.jsonl)

Found alpaca dataset, so started with that instead of self_instruct


## How to run the pipeline


---
### 0. Pre-requisites


0. Requires `Python3.9` and `sqlite3`
1. Download dataset from hub using the following command:
  - Flan
```
sudo apt-get install git-lfs  # if you don't have git lfs
git lfs install
git clone https://huggingface.co/datasets/Muennighoff/flan
```
  - Alpaca
```
wget "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
```
2. Install the requirements
```
pip install -r requirements.txt
```
3. Set the following environment variables:
```
export FLAN_DATA_DIR=<path to the flan dataset>
export DB_PATH=<path to the sqlite db>   # DB will be created by the loader script
```


---
### 1. Load the dataset in a DB


```
python src/loaders/flan_loader.py
#or for alpaca
python src/loaders/alpaca_loader.py
```


- all: loads all the datasets
- dataset_name: loads the dataset with the name dataset_name. For available datasets, check the Hub


---
### 2. Train a model in supervised-finetuning mode


```
TOKENIZERS_PARALLELISM=true \
python3 run_sft_experiment.py \
    --dataset alpaca \
    --train_size -1 \
    --val_size -1 \
    --decoder_only \
    --model 'EleutherAI/gpt-neo-1.3B' \
    --accelerator 'cuda' \
    --devices 1 \
    --num_workers 6 \
    --strategy "deepspeed_stage_3" \
    --precision 16 \
    --max_len 2048 \
    --train_steps 36000 \
    --val_steps 2000 \
    --val_freq 5000 \
    --train_batch_size_per_device 4 \
    --val_batch_size_per_device 8 \
    --lr 0.00002 \
    --grad_accumulate_batches 32 \
    --lr_update_freq 800 \
    --warmup \
    --save_freq 100 \
    --save_dir './models/v5/' \
;
```


<b>Note:</b> Because we use deepspeed stage 3, it saves the a checkpoint directory for each epoch. We need to collate them into a single one. To do that, go to the appropriate checkpoint directory and run the following command:
```
python src/zero_to_pytorch.py  --model "distilgpt2" --save_dir "./models/v1/" --ckpt_dir_name "best.ckpt" --model_name "pytorch_model.bin"
```


Currently, the trainer only supports Single-Node Multi-GPU training. Multi-Node training is on Todos.


---
### 3. Generate output for text prompts


---
### 4. Evaluation Pipeline 


---
# Model Versions

- Before v4: [Completed]
    - Testing model pipelines
    - Crude version
    - experimental

- v4: [Completed]
    - Base Model: EleutherAI/gpt-neo-1.3B
    - Dataset: Alpaca
    - Steps: 18000 (Approx 1.5 epochs)
    - First time model training completed, so the model is not tuned enough
    - Remarks:
        - Generates pretty good text
        - Is able to follow instructions mostly one instruction only
        - struggles with chain of instructions
    - Evaluations:
        - [ADD HERE]

- v5: [Running]
    - Base Model: EleutherAI/gpt-neo-1.3B
    - Dataset: Alpaca
    - Steps: 36000 (Approx 3 epochs)
    - Gradient Accumulation upto total batch size of 128
    - Remarks:
        - [ADD HERE]
    - Evaluations:
        - [ADD HERE]

- v6: [TODO]
    - Base Model: EleutherAI/gpt-neo-1.3B
    - Dataset: Alpaca
    - Steps: 36000 (Approx 3 epochs)
    - WandB Run Path: [ADD HERE]
    - Parameter Efficient Fine-tuning
        - As far as I understand it is just freezing last layers and only fine-tune final layers
    - Remarks:
        - [ADD HERE]
    - Evaluations:
        - [ADD HERE]

- v7: [TODO]
    - Base Model: EleutherAI/gpt-neo-1.3B
    - Dataset: Alpaca
    - Steps: 36000 (Approx 3 epochs)
    - Concatenating the individual examples together to utilize the entire context len available
      and it might also help in chain of instructions
    - Remarks:
        - [ADD HERE]
    - Evaluations:
        - [ADD HERE]

---
### Todos:
- [x] Add a script to load the dataset in a DB
- [x] Add main function to dataset creator to create idxs
- [x] Collate the deepspeed model checkpoints into a single one
- [x] Add FusedAdam or DeepSpeedCPUAdam optimizer for deepspeed strategy
- [x] Add a line for defining the strategy if it is DeepSpeed
- [x] Train a model in supervised-finetuning mode
- [ ] Evaluation pipeline
- [ ] Add support for PEFT
- [ ] Add support for Multi-Node training (Distributed Training)
- [x] Generate output for text prompts
- [ ] Create Labeling instructions (Checkout InstructGPT doc)
- [ ] Label the generated output using Human
- [ ] Finetune a reward model using the labels provided by the human
- [ ] Train the model in RLHF fashion using PPO limits
