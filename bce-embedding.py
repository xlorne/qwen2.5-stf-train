import os
import torch
from transformers import AutoModel, AutoTokenizer,AutoModelForSequenceClassification
# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 本地的知识库数据
sentences = ['sentence_0', 'sentence_1']

# init model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('maidalun1020/bce-embedding-base_v1')
model = AutoModel.from_pretrained('maidalun1020/bce-embedding-base_v1')

model.to(device)

# get inputs
inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

# get embeddings
outputs = model(**inputs_on_device, return_dict=True)
embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize

# 向量化后的数据
print(embeddings)

# # init model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('maidalun1020/bce-reranker-base_v1')
model = AutoModelForSequenceClassification.from_pretrained('maidalun1020/bce-reranker-base_v1')

model.to(device)

# 用户的问题
query = 'input_query'
# 本地的知识库向量检索后的数据
passages = ['passage_0', 'passage_1']

sentence_pairs = [[query, passage] for passage in passages]

# get inputs
inputs = tokenizer(sentence_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

# calculate scores
scores = model(**inputs_on_device, return_dict=True).logits.view(-1,).float()
scores = torch.sigmoid(scores)

## 获取最相近的答案排名
print(scores)



