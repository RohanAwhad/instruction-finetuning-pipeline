source dev.env

python3 run_sft_experiment.py \
    --dataset alpaca \
    --train_size 200 \
    --val_size 20 \
    --decoder_only \
    --accelerator 'cpu' \
    --devices 1 \
    --num_workers 0 \
    --precision 32 \
    --max_len 1024 \
    --train_steps 50 \
    --val_steps 10 \
    --val_freq 25 \
    --train_batch_size_per_device 2 \
    --val_batch_size_per_device 4 \
    --lr_update_freq 5 \
    --save_dir './models/test_run/' \
;
