import gc
from datetime import datetime
from typing import Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.trainer_utils import set_seed

# 设置随机种子以确保可重现性
set_seed(42)


class QwenRunner:
    def __init__(
        self,
        base_model_path: str,
        device: str = None,
        max_new_tokens: int = 512,
        top_p: float = 0.8,
        temperature: float = 0.8
    ):
        """
        初始化 QwenRunner

        Args:
            base_model_path: 基座模型路径
            device: 运行设备，默认自动检测
            max_new_tokens: 最大生成长度
            top_p: top-p 采样参数
            temperature: 温度参数
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else
                               "mps" if torch.backends.mps.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.temperature = temperature

        # 加载基座模型和分词器
        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map={"": self.device},
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32  # 对CUDA使用FP16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )

        # 初始化当前活跃的 LoRA 模型
        self.current_lora_path = None
        self.model = None
        self.pipe = None

    def load_lora(self, lora_model_path: str):
        """
        加载新的 LoRA 模型

        Args:
            lora_model_path: LoRA 模型路径
        """
        # 如果要加载的模型与当前模型相同，直接返回
        if self.current_lora_path == lora_model_path:
            return

        # 清理当前模型
        if self.model is not None:
            del self.model
            del self.pipe
            torch.cuda.empty_cache()
            gc.collect()

        # 加载新的 LoRA 模型
        self.model = PeftModel.from_pretrained(
            self.base_model,
            lora_model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32  # 对CUDA使用FP16
        )

        # 创建新的 pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32  # 对CUDA使用FP16
        )

        self.current_lora_path = lora_model_path

    def run(self, question: str):
        """
        运行模型推理

        Args:
            question: 输入问题
        Returns:
            str: 模型回答
        """
        if self.model is None:
            raise RuntimeError("No LoRA model loaded. Please call load_lora() first.")

        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

        try:
            with torch.cuda.amp.autocast() if self.device == "cuda" else nullcontext():  # 使用AMP
                result = self.pipe(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    do_sample=True,
                    num_return_sequences=1
                )

            output = result[0]['generated_text']
            assistant_responses = output.split("<|im_start|>assistant\n")
            if len(assistant_responses) > 1:
                response = assistant_responses[-1].split("<|im_end|>")[0].strip()
            else:
                response = output.split("<|im_end|>")[0].strip()

            return response

        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return f"推理错误: {str(e)}"

    def __del__(self):
        """清理资源"""
        if self.model is not None:
            del self.model
        if self.pipe is not None:
            del self.pipe
        torch.cuda.empty_cache()
        gc.collect()


class nullcontext:
    """一个简单的空上下文管理器"""
    def __enter__(self): pass
    def __exit__(self, *args): pass


def main():
    print("\nTesting optimized model with multiple LoRA weights...")

    base_model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_model_map = {
        "bookmodel": "./qwen_sft_bookmodel",
        "tvmodel": "./qwen_sft_tvmodel"
    }

    runner = QwenRunner(
        base_model_path,
        temperature=0.8,
        top_p=0.9,
        max_new_tokens=512
    )

    test_questions = [
        {
            "model": "tvmodel",
            "question": "声音太大了，调小一些？"
        },
        {
            "model": "bookmodel",
            "question": "帮我查看一下王亮的机票信息，编号为1002？"
        },
        {
            "model": "tvmodel",
            "question": "我想要看周星驰的电影？"
        },
        {
            "model": "tvmodel",
            "question": "声音太小了，帮我调大一些？"
        },
        {
            "model": "bookmodel",
            "question": "帮我修改章三的机票日期，调整到2013-1-22，订单编号是1002"
        }
    ]

    for item in test_questions:
        question = item["question"]
        model_name = item["model"]
        t1 = datetime.now().timestamp() * 1000
        runner.load_lora(lora_model_map[model_name])
        answer = runner.run(question)
        t2 = datetime.now().timestamp() * 1000
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"Time: {t2 - t1:.2f}ms")


if __name__ == "__main__":
    main()