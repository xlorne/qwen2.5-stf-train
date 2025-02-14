import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    # 测试模型
    print("\nTesting fine-tuned model...")
    test_model = AutoModelForCausalLM.from_pretrained(
        "./qwen_sft_model",
        device_map={"": device},
        trust_remote_code=True
    )
    test_tokenizer = AutoTokenizer.from_pretrained(
        "./qwen_sft_model",
        trust_remote_code=True
    )

    # 测试所有问题
    test_questions = [
        "声音太大了，调小一些？",
        "周杰伦最近有什么音乐？",
        "我想要看周星驰的电影？",
        "声音太小了，帮我调大一些？",
        "我想要听一首王力宏的歌曲？",
    ]

    pipe = pipeline("text-generation", model=test_model, tokenizer=test_tokenizer)

    print("\nTesting with all questions:")
    for question in test_questions:
        print(f"\nQ: {question}")
        prompt = f"<|im_start|>system\n你是一个电视的控制器，识别用户的行为响应控制指令.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        result = pipe(prompt, max_new_tokens=512, top_p=0.7, temperature=0.95)
        print(f"A: {result[0]['generated_text']}")


if __name__ == "__main__":
    main()
