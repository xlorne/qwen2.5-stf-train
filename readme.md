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

## Load model to ollama
* Edit model ModelFile https://github.com/ollama/ollama
```
cat ./tv-model-qwen2.5-0.5b.modelfile
```
* load local modelfile to ollama
```
ollama create tv-model -f tv-model-qwen2.5-0.5b.modelfile
```

* show ollama model
```
ollama list
```

* test model with curl
```
curl --location 'http://localhost:11434/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [         
        {
            "content": [
                {
                    "type": "text",
                    "text": "播放一下周杰伦的音乐"
                }
            ],
            "role": "user"
        }
    ],
    "model": "tv-model:latest",
    "stream": false,
    "temperature": 0.1
}'
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