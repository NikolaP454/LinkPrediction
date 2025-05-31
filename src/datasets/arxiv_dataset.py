import torch
from torch_geometric.data import Data


class ArxivDataset:
    def __init__(self, data: Data):
        self.titles = data.x[:, 0]
        self.abstracts = data.x[:, 1]

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
            "id": idx,
            "title": self.titles[idx],
            "abstract": self.abstracts[idx],
            "cites": self.data.edge_index[1][self.data.edge_index[0] == idx].tolist(),
            "is_cited_by": self.data.edge_index[0][
                self.data.edge_index[1] == idx
            ].tolist(),
        }

    def get_data(self):
        return self.data

    def get_title(self, idx):
        return self.titles[idx]

    def get_abstract(self, idx):
        return self.abstracts[idx]

    def get_cites(self, idx, ignore: int = -1):
        cites_node_ids = self.data.edge_index[1][
            self.data.edge_index[0] == idx
        ].tolist()

        if ignore != -1:
            cites_node_ids = [c for c in cites_node_ids if c != ignore]

        return [self.titles[cite] for cite in cites_node_ids if cite < len(self.titles)]

    def get_is_cited_by(self, idx, ignore: int = -1):
        is_cited_by_node_ids = self.data.edge_index[0][
            self.data.edge_index[1] == idx
        ].tolist()

        if ignore != -1:
            is_cited_by_node_ids = [c for c in is_cited_by_node_ids if c != ignore]

        return [
            self.titles[cited_by]
            for cited_by in is_cited_by_node_ids
            if cited_by < len(self.titles)
        ]
