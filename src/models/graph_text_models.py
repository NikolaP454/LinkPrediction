import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from .text_embedding_models import TextEmbedding


class SageConvModel(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 64,
        out_channels: int = 128,
        use_titles: bool = True,
        use_abstracts: bool = True,
        reduced_dim_titles: int = 0,
        reduced_dim_abstracts: int = 0,
    ):
        super(SageConvModel, self).__init__()

        # Base configurations
        self.use_titles = use_titles
        self.use_abstracts = use_abstracts
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"

        # Initialize the text embedding models based on the flags
        assert (
            use_titles or use_abstracts
        ), "At least one of titles or abstracts must be used."

        if use_titles:
            self.title_embedder = TextEmbedding(out_dimension=reduced_dim_titles)

        if use_abstracts:
            self.abstract_embedder = TextEmbedding(out_dimension=reduced_dim_abstracts)

        # Calculate the input dimension based on the embeddings used
        # 768 for BERT embedding size
        graph_x_dim = 0

        if use_titles:
            graph_x_dim += reduced_dim_titles if reduced_dim_titles > 0 else 768

        if use_abstracts:
            graph_x_dim += reduced_dim_abstracts if reduced_dim_abstracts > 0 else 768

        # Initialize the graph model
        self.conv1 = SAGEConv(graph_x_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

        self.classifier = nn.Linear(out_channels * 2, 1)

    def __joint_embeddings(self, embedded_titles, embedded_abstracts):
        if self.use_titles and self.use_abstracts:
            return torch.cat([embedded_titles, embedded_abstracts], dim=1)

        elif self.use_titles:
            return embedded_titles

        elif self.use_abstracts:
            return embedded_abstracts

        raise ValueError(
            "At least one of titles or abstracts must be used. (Joint Embedding Error)"
        )

    def single_node_embedding(self, x_titles_inputs, x_abstracts_inputs, edge_index):
        # Embed titles and abstracts if they are used
        embedded_titles, embedded_abstracts = None, None

        if self.use_titles:
            embedded_titles = self.title_embedder(x_titles_inputs, True).to(self.device)

        if self.use_abstracts:
            embedded_abstracts = self.abstract_embedder(x_abstracts_inputs).to(
                self.device
            )

        # Combine embeddings based on the flags
        x_combined = self.__joint_embeddings(embedded_titles, embedded_abstracts)

        # Embed the nodes using SAGEConv
        x = self.conv1(x_combined, edge_index).relu()
        x = self.conv2(x, edge_index)

        return x

    def forward(
        self, x_titles_inputs, x_abstracts_inputs, edge_index, edge_label_index
    ):
        embeddings = self.single_node_embedding(
            x_titles_inputs, x_abstracts_inputs, edge_index
        )

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
        self.device = device

        self.title_embedder.to(device) if self.use_titles else None
        self.abstract_embedder.to(device) if self.use_abstracts else None

        self.conv1.to(device)
        self.conv2.to(device)

        self.classifier.to(device)

        return self
