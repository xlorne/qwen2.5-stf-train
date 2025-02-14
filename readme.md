# Qwen2.5B STF training on MSP

## Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install modelscope accelerate transformers peft datasets addict
```

## FQA
1. download huggingface model fail
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

2. Specified GPU device 
```
export CUDA_VISIBLE_DEVICES=0,1,2
```