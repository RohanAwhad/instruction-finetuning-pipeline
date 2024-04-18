# LR and Batch Size are based on the Alpaca paper
# turning on gradient accumulation, increments global step 
# post that many steps

# because of the adventures in gradient accumulation in pytorch lightning
# have to make a few calculated changes to this!
# I have a dataset with approx 52K samples
# 90% will be used for training i.e. 46800
# I have per device batch size of 4
# I want a total batch size for each step to be 128
# that makes grad accumulate batches = 128 / (4 * n_devices) = 32
# Now with a batch size of 128, my train_steps for 1 epoch become => 366
# I want 3 epochs, so my total train steps = 366 * 3 = 1098

# gradient accumulation doesn't hamper val steps
# but I have to adjust my val_freq
# If I want 2 validations per epoch,
# then my val_freq = (46800 / (4 * n_devices)) / 2 => 5850

# LR Update Freq: this again follows regulars steps, and not the gradient accumulation steps
# so if I want to update my lr post every step, because the batch size is large enough,
# I can just say freq to be 32

# Save freq relies on train steps, so that should be in accordance with it. 100 sounds appropriate

# TODO (rohan): bring these calculations in the script and make it simpler to run this script
# no need to work like this

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
    --train_steps 1098 \
    --val_steps 2000 \
    --val_freq 5850 \
    --train_batch_size_per_device 4 \
    --val_batch_size_per_device 8 \
    --lr 0.00002 \
    --grad_accumulate_batches 32 \
    --lr_update_freq 32 \
    --warmup \
    --save_freq 100 \
    --save_dir './models/v5/' \
;
