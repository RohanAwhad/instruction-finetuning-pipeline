source dev.env

# TODO (rohan): Should move this cal to file
# LR is calculated based on Alpaca paper
# They used 2e-5 for 128 batch size
# Calculated for this using following formula:
# 2e-5 * (batch_size / 128)

python3 run_sft_experiment.py \
    --dataset alpaca \
    --train_size 2400 \
    --val_size 240 \
    --decoder_only \
    --model 'EleutherAI/gpt-neo-125M' \
    --accelerator 'cuda' \
    --devices 2 \
    --num_workers 2 \
    --strategy "deepspeed_stage_3" \
    --precision 16 \
    --lr 0.0000075 \
    --max_len 1024 \
    --train_steps 50 \
    --val_steps 10 \
    --val_freq 10 \
    --train_batch_size_per_device 6 \
    --val_batch_size_per_device 12 \
    --lr_update_freq 16 \
    --warmup \
    --grad_accumulate_batches 16 \
    --save_freq 4 \
    --save_dir './models/test_2/' \
;
