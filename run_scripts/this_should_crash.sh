source dev.env

# This should not crash now. it was for testing only

# TODO (rohan): Should move this cal to file
# LR is calculated based on Alpaca paper
# They used 2e-5 for 128 batch size
# Calculated for this using following formula:
# 2e-5 * (batch_size / 128)

python3 run_sft_experiment.py \
    --dataset alpaca \
    --train_size -1 \
    --val_size -1 \
    --decoder_only \
    --model 'EleutherAI/gpt-neo-125M' \
    --accelerator 'cuda' \
    --devices 2 \
    --num_workers 2 \
    --strategy "deepspeed_stage_3" \
    --precision 16 \
    --lr 0.0000075 \
    --max_len 1024 \
    --train_steps 7000 \
    --val_steps 100 \
    --val_freq 3000 \
    --train_batch_size_per_device 6 \
    --val_batch_size_per_device 12 \
    --lr_update_freq 10 \
    --save_dir './models/test_run_gpu_alpaca/' \
;
