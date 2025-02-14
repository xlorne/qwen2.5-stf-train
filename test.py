import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from transformers.trainer_utils import set_seed

# 设置随机种子以确保可重现性
set_seed(42)

class QwenRunner:
    def __init__(self,
                 model_path,
                 max_new_tokens=512,
                 top_p=0.8,
                 temperature=0.8):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.temperature = temperature

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": self.device},
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def run(self, question):
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        result = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        output = result[0]['generated_text']
        return output.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]


def main():
    # 测试模型
    print("\nTesting fine-tuned model...")

    model_path = "./qwen_sft_model"

    model = QwenRunner(
        model_path,
        temperature=0.8,
        top_p=0.9,
        max_new_tokens=512
    )

    # 测试所有问题
    test_questions = [
        "声音太大了，调小一些？",
        "周杰伦最近有什么音乐？",
        "我想要看周星驰的电影？",
        "声音太小了，帮我调大一些？",
        "我想要听一首王力宏的歌曲？",
    ]

    print("\nTesting with all questions:")
    for question in test_questions:
        print(f"\nQ: {question}")
        print(f"A: {model.run(question)}")


if __name__ == "__main__":
    main()
