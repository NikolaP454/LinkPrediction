import torch
from torch_geometric.data import Data

from ..models import TextTokenizer


def tokenize_data(data: Data, is_train: bool) -> Data:
    """
    Tokenizes the text data in the PyTorch Geometric Data object.

    Args:
        data (Data): The PyTorch Geometric Data object containing text data.
        is_train (bool): Whether the data is for training or not.

    Returns:
        Data: The updated Data object with tokenized text.
    """
    tokenizer = TextTokenizer()

    titles = list(data.x[:, 0])
    abstracts = list(data.x[:, 1])

    tokenized_titles = tokenizer(titles)
    tokenized_abstracts = tokenizer(abstracts)

    x = torch.stack((tokenized_titles, tokenized_abstracts), dim=1).contiguous()

    if is_train:
        return Data(
            x=x,
            edge_index=data.edge_index.contiguous(),
        )

    return Data(
        x=x,
        edge_index=data.edge_index.contiguous(),
        edge_label=data.edge_label.contiguous(),
    )
