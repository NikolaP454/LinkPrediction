import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import SAGEConv

from .text_embedding_models import TextEmbedding


class SageConvModel(nn.Module):
    def __init__(self, in_channels=0, hidden_channels=64, out_channels=128):
        super(SageConvModel, self).__init__()
        self.text_embedding = TextEmbedding()

        # 768 for BERT embedding size
        self.conv1 = SAGEConv(in_channels + 768 * 2, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

        self.classifier = nn.Linear(out_channels * 2, 1)

    def single_node_embedding(self, x_titles, x_abstracts, edge_index):
        embedded_titles = self.text_embedding(x_titles)
        embedded_abstracts = self.text_embedding(x_abstracts)

        x_combined = torch.cat([embedded_titles, embedded_abstracts], dim=1)

        x = self.conv1(x_combined, edge_index).relu()
        x = self.conv2(x, edge_index)

        return x

    def forward(self, x_titles, x_abstracts, edge_index, edge_label_index):
        embeddings = self.single_node_embedding(x_titles, x_abstracts, edge_index)

        u_embeddings = torch.index_select(embeddings, 0, edge_label_index[0])
        v_embeddings = torch.index_select(embeddings, 0, edge_label_index[1])

        total_x = torch.cat([u_embeddings, v_embeddings], dim=1)

        return self.classifier(total_x)

    def load_pretrained(self, model_path, device="cpu"):
        state_dict = torch.load(model_path, map_location=torch.device(device))
        self.load_state_dict(state_dict)

    def save_pretrained(self, model_path):
        torch.save(self.state_dict(), model_path)

    def to(self, device):
        super(SageConvModel, self).to(device)

        self.text_embedding.to(device)
        self.conv1.to(device)
        self.conv2.to(device)
        self.classifier.to(device)

        return self
