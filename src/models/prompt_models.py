import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification

from .text_embedding_models import TextTokenizer


class PromptLinkPredictionModel(nn.Module):
    def __init__(
        self, model_name: str = "distilbert-base-uncased", max_length: int = 512
    ):
        super(PromptLinkPredictionModel, self).__init__()

        # Initialize the tokenizer and model
        self.tokenizer = TextTokenizer(model_name=model_name, max_length=max_length)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )

        self.output = nn.Sigmoid()

        # Other initializations
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, texts):
        inputs = self.tokenizer(texts).to(self.device)
        outputs = self.model(**inputs)

        return self.output(outputs.logits)

    def load_pretrained(self, model_path, device="cpu"):
        state_dict = torch.load(model_path, map_location=torch.device(device))
        self.load_state_dict(state_dict)

        self.to(device)

        print(f"Model loaded from {model_path} on device {device}")

    def to(self, device):
        super(PromptLinkPredictionModel, self).to(device)

        self.device = device
        self.model.to(device)
        self.output.to(device)

        return self
