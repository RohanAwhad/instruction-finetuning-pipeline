source dev.env

python3 run_sft_experiment.py \
    --dry_run \
    --find_batch_size \
    --dataset alpaca \
    --train_size 200 \
    --decoder_only \
    --accelerator 'cpu' \
    --num_workers 0 \
    --devices 1 \
    --precision 16 \
    --max_len 1024 \
    --train_steps 10 \
    --val_steps 10 \
    --val_freq 10 \
    --train_batch_size_per_device 1 \
    --val_batch_size_per_device 1 \
    --lr_update_freq 10 \
    --save_dir './models/dry_run/' \
;

