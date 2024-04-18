mkdir -p ./models/testing_onnx/quantized
python3 src/export_to_onnx.py \
    --model_config /Users/rohan/3\ Resources/testing_models/tiny-random-gpt_neo \
    --model_path_fp32 ./models/testing_onnx/model.onnx \
    --model_path_quantized ./models/testing_onnx/quantized/model.onnx \
    --max_len 100;
