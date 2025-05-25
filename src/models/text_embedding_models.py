import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel


class TextEmbedding(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(TextEmbedding, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)

    def forward(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]

        return embeddings

    def to(self, device):
        super(TextEmbedding, self).to(device)

        self.bert_model.to(device)
        self.tokenizer = self.tokenizer.to(device)

        return self
