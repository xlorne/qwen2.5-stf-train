import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import set_seed

# 设置随机种子以确保可重现性
set_seed(42)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def verify_dataset(file_path="train.json"):
    """验证数据集格式并打印示例"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\nDataset contains {len(data)} examples")
    print("\nFirst example transformed format:")
    example = data[0]
    formatted = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
    print(formatted)
    return data


def prepare_dataset(tokenizer, data_file="train.json"):
    """准备训练数据集"""
    dataset = load_dataset("json", data_files={"train": data_file})

    def preprocess_function(examples):
        inputs = [f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>" for
                  instruction, output in zip(examples["instruction"], examples["output"])]
        model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=2048)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    print("\nProcessing dataset...")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=16
    )
    print(f"Processed {len(tokenized_datasets['train'])} examples")
    return tokenized_datasets["train"]



def main():
    print("Starting training process...")

    # 验证数据集
    verify_dataset()

    # 初始化模型和tokenizer
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\nLoading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": device},
        trust_remote_code=True,
    )

    # 准备数据集
    train_dataset = prepare_dataset(tokenizer)

    # 配置LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["gate_proj", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 训练配置
    training_args = TrainingArguments(
        output_dir="./qwen_sft",
        per_device_train_batch_size=2,  # 减小批量大小
        gradient_accumulation_steps=16,  # 增加梯度累积步数
        num_train_epochs=3,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",  # 使用 cosine 学习率调度器
        warmup_steps=0,
        max_grad_norm=1.0,  # 添加梯度裁剪
        logging_steps=5,
        save_steps=100,
        eval_strategy="no",
        bf16=True,  # 启用 bf16
        optim="adamw_torch",  # 使用 adamw_torch 优化器
        report_to="none",
        fp16=False,
    )

    # 开始训练
    print("\nStarting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    # 保存模型
    print("\nSaving model...")
    model.save_pretrained("./qwen_sft_model")
    tokenizer.save_pretrained("./qwen_sft_model")

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