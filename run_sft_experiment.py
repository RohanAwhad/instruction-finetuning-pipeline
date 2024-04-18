import argparse
import os
import pytorch_lightning as pl
import random
import time
import torch
import torch.nn.functional as F
import torch.profiler
import wandb

from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, PreTrainedTokenizer

from src.db_model import DataInstance
from src.dataset import SFTDataset
from src.lightning_data_module import SFTDataModule
from src.lightning_ml_module import LightningLanguageModel

# (ref): https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul_allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# recommended by lightning to utilize tensor cores
torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":
    # arg parsers
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train_size', type=int, default=-1) 
    parser.add_argument('--val_size', type=int, default=-1) 
    parser.add_argument("--decoder_only", action='store_true')
    parser.add_argument('--model', type=str, default='distilgpt2')

    parser.add_argument("--profiler", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument('--find_batch_size', action='store_true', help='Use to set seqlen to max for all input to figure out batch size for worst case scenario')

    parser.add_argument("--accelerator", type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--devices", type=int)

    parser.add_argument("--strategy", type=str)
    parser.add_argument("--precision", type=str)   
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--lr', type=float, default=9.65e-6)
    parser.add_argument('--warmup', action='store_true')

    parser.add_argument("--train_steps", type=int)
    parser.add_argument("--val_steps", type=int)
    parser.add_argument("--val_freq", type=int)
    parser.add_argument("--train_batch_size_per_device", type=int)
    parser.add_argument("--val_batch_size_per_device", type=int)
    parser.add_argument('--grad_accumulate_batches', type=int, default=1)
    parser.add_argument("--lr_update_freq", type=int)

    parser.add_argument('--save_freq', type=int)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print('Creating Data Module ... ')
    # === dataset splitting ===
    dataset = DataInstance.select().where(DataInstance.dataset == args.dataset)
    dataset_size = len(dataset)
    if args.train_size == -1:
        train_size = int(dataset_size * 0.9)  # 90% of data is used for training
        val_size = dataset_size - train_size
    else:
        train_size = args.train_size
        val_size = args.val_size

    dataset_ids = list(range(dataset_size))
    random.seed(1)
    random.shuffle(dataset_ids)
    train_ids = dataset_ids[:train_size]
    val_ids = dataset_ids[-val_size:]
    # ===

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.find_batch_size:
        data = [' '.join((x.input, x.target)) for x in dataset]
        data_tokenized_len = [len(tokenizer.encode(x)) for x in data]
        args.max_len = min(args.max_len, max(data_tokenized_len))
        print('Setting max len to:', args.max_len)

    # === Data Module === 
    sft_train_dataset = SFTDataset(train_ids, tokenizer, dataset, max_len=args.max_len, find_batch_size=args.find_batch_size)
    sft_val_dataset = SFTDataset(val_ids, tokenizer, dataset, max_len=args.max_len, find_batch_size=args.find_batch_size)

    data_module = SFTDataModule(
        sft_train_dataset,
        val_dataset=sft_val_dataset,
        eos_token_id=tokenizer.eos_token_id,
        train_batch_size=args.train_batch_size_per_device,
        val_batch_size=args.val_batch_size_per_device,
        num_workers=args.num_workers,
    )

    # ==== 
    print('Data Module creation complete!')

    if args.decoder_only: base_model = AutoModelForCausalLM.from_pretrained(args.model)
    else: base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    scheduler_t_max = args.train_steps // args.lr_update_freq
    step_size = args.train_batch_size_per_device * args.devices
    if args.grad_accumulate_batches is not None:
        step_size *= args.grad_accumulate_batches
    n_total_iter = args.train_steps * step_size
    
    model_config = dict(
        model=base_model,
        step_size=step_size,
        n_warmup=int(n_total_iter // 10) if args.warmup else 0,
        n_total_iter=n_total_iter,
        lr=args.lr,
        update_every_n_steps=args.lr_update_freq,
        strategy=args.strategy
    )
    model = LightningLanguageModel(**model_config)

    # logging & callbacks
    loggers, callbacks = [], []
    if not args.dry_run:
        loggers.append(WandbLogger(project="SFT for IFT Pipeline"))
        callbacks.extend((
            ModelCheckpoint(dirpath=args.save_dir, save_last=False, every_n_train_steps=args.save_freq),
            LearningRateMonitor(logging_interval='step'),
        ))
    elif (args.dry_run or args.profiler):
        loggers.append(TensorBoardLogger(args.save_dir))
        callbacks.append(DeviceStatsMonitor(cpu_stats=True))

    if args.profiler == 'pytorch':
        args.profiler = PyTorchProfiler(
            dirpath=os.path.join(args.save_dir, 'pytorch_profiler_logs'),
            filename=str(time.monotonic()),
            profile_memory=True,
            schedule=torch.profiler.schedule(skip_first=150, wait=5, warmup=2, active=5, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(loggers[0].log_dir),
        )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_steps=args.train_steps,
        precision=args.precision,
        fast_dev_run=args.dry_run,
        gradient_clip_val = 1.0,
        gradient_clip_algorithm='norm',
        accumulate_grad_batches=args.grad_accumulate_batches,
        # val
        check_val_every_n_epoch=None,
        val_check_interval=args.val_freq,
        limit_val_batches=args.val_steps,
        # logging & callbacks
        profiler=args.profiler,
        strategy=args.strategy,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1 if args.grad_accumulate_batches > 1 else 50
    )
    trainer.fit(model, data_module)
