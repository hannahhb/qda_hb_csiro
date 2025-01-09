
# python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.3 --port 8000 --tokenizer-mode mistral \
#     --dtype bfloat16

CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 \
    --model-path  mistralai/Mistral-7B-Instruct-v0.3 \
    --dtype bfloat16
