from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基础模型
base_model_name = "Qwen/Qwen2.5-0.5B"  # 假设您使用的是0.5B版本，如有不同请调整
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 加载adapter
adapter_path = "./qwen_sft_model/"
model = PeftModel.from_pretrained(base_model, adapter_path)

# 合并模型
merged_model = model.merge_and_unload()

# 保存完整模型
output_dir = "./qwen_sft_merged/"
merged_model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
