import torch
from torch_geometric.data import Data


class ArxivDataset:
    def __init__(self, data: Data):
        self.data = Data(
            x=data.x,
            edge_index=data.edge_index,
        )

        # Just for the sake of having a consistent data structure
        self.data.x = self.data.x.contiguous()
        self.data.edge_index = self.data.edge_index.contiguous()

    def __len__(self):
        return len(self.data.x)

    def __getitem__(self, idx):
        return {
            "title-inputs": self.data.x[:, 0][idx],
            "abstract-inputs": self.data.x[:, 1][idx],
            "edge_index": self.data.edge_index,
        }

    def get_data(self):
        return self.data

    def get_title(self, idx):
        return self.data.x[:, 0][idx]

    def get_abstract(self, idx):
        return self.data.x[:, 1][idx]
