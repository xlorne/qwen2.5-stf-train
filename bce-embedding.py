import torch
from transformers import AutoModel, AutoTokenizer,AutoModelForSequenceClassification

# list of sentences
sentences = ['sentence_0', 'sentence_1', ...]

# init model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('maidalun1020/bce-embedding-base_v1')
model = AutoModel.from_pretrained('maidalun1020/bce-embedding-base_v1')

device = 'cpu'  # if no GPU, set "cpu"
model.to(device)

# get inputs
inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

# get embeddings
outputs = model(**inputs_on_device, return_dict=True)
embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize
print(embeddings)
#
#
#
# # init model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained('maidalun1020/bce-reranker-base_v1')
# model = AutoModelForSequenceClassification.from_pretrained('maidalun1020/bce-reranker-base_v1')
#
# device = 'cpu'  # if no GPU, set "cpu"
# model.to(device)
#
# # get inputs
# inputs = tokenizer(sentence_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
# inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
#
# # calculate scores
# scores = model(**inputs_on_device, return_dict=True).logits.view(-1,).float()
# scores = torch.sigmoid(scores)
