(rohan 22/03/2023)

```
# for dealing with OOM
# (ref): https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
# for variable input size this will hurt the performance, because it acts more like JIT with fixed shape
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.enabled = True

# (ref): https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul_allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# recommended by lightning to utilize tensor cores
torch.set_float32_matmul_precision('medium')

# setting precision to bf16 doesn't seem to give any benefit, but in turn hamper the performance

```

(rohan 22/03/2023)

- When using Gradient Accumulation in pytorch lightning, train_steps refer to
  batch_size_per_device * n_devices * accumulate_grad_batches
- But the sad part is lr_update follows batch_size_per_device * n_devices
- Even sadder is val_freq also follow the same
- WTF is happening with pytorch lightning
