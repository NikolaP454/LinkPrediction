import torch
import torch.nn as nn

from transformers import DistilBertTokenizer, DistilBertModel


class TextEmbedding(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super(TextEmbedding, self).__init__()

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.bert_model = DistilBertModel.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(self.device)

        outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]

        return embeddings

    def to(self, device):
        super(TextEmbedding, self).to(device)
        self.device = device

        self.bert_model.to(device)
        return self
