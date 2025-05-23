import torch
from torch_geometric.data import Data


class ArxivDataset:
    def __init__(self, data: Data):

        self.titles = list(data.x[:, 0])
        self.abstracts = list(data.x[:, 1])

        self.data = Data(
            x=torch.Tensor(range(len(self.titles))),
            edge_index=data.edge_index,
        )

        # Just for the sake of having a consistent data structure
        self.data.x = self.data.x.contiguous()
        self.data.edge_index = self.data.edge_index.contiguous()

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        return {
            "title": self.titles[idx],
            "abstract": self.abstracts[idx],
            "edge_index": self.data.edge_index,
        }

    def get_data(self):
        return self.data

    def get_title(self, idx):
        return self.titles[idx]

    def get_abstract(self, idx):
        return self.abstracts[idx]
