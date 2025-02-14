# Qwen2.5B STF training on MSP

## Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install modelscope accelerate transformers peft datasets addict sentencepiece
```

## Export GGUF model
* merge stf model and qwen model, and save it to qwen_sft_merged 
```
python marge.py
```
* install llama.cpp on your machine, download https://github.com/ggerganov/llama.cpp/releases binary and add it to your PATH
* clone llama.cpp code `git clone git@github.com:ggerganov/llama.cpp.git`
* cd llama.cpp and convert model 
```
cd llama.cpp
python convert_hf_to_gguf.py /Users/lorne/developer/github/python/qwen2.5-stf/qwen_sft_merged --outfile qwen2.5-0.5B-q8_0.gguf --outtype q8_0
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