import torch
import torch.nn as nn

from transformers import DistilBertTokenizer, DistilBertModel


class TextEmbedding(nn.Module):
    def __init__(
        self, model_name: str = "distilbert-base-uncased", out_dimension: int = 0
    ):
        super(TextEmbedding, self).__init__()

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.bert_model = DistilBertModel.from_pretrained(model_name)

        # If out_dimension is specified, we add a linear layer to reduce the output dimension
        if out_dimension > 0:
            self.reducer = nn.Linear(self.bert_model.config.hidden_size, out_dimension)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(self.device)

        outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]

        if hasattr(self, "reducer"):
            embeddings = self.reducer(embeddings)

        return embeddings

    def to(self, device):
        super(TextEmbedding, self).to(device)
        self.device = device

        self.bert_model.to(device)

        if hasattr(self, "reducer"):
            self.reducer.to(device)

        return self
