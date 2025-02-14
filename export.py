import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "./qwen_sft_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 执行动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # 量化特定层类型（例如Linear层）
    dtype=torch.qint8   # 选择量化的位数（8位量化）
)


quantized_model.save_pretrained('./qwen2.5_0.5B_INT8')