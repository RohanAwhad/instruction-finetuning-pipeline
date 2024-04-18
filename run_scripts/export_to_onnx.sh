python3 src/export_to_onnx.py \
    --model_config EleutherAI/gpt-neo-1.3B \
    --trained_model_path ./models/v5/gpt-neo-1.3B-3epochs.bin \
    --model_path_fp32 ./models/v5/gpt-neo-1.3B-3epochs.onnx \
    --model_path_quantized ./models/v5/gpt-neo-1.3B-3epochs-quantized/model.onnx \
    --max_len 2048;
