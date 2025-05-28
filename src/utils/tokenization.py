import torch
from torch_geometric.data import Data

from ..models import TextTokenizer


def tokenize_data(
    data: Data,
    is_train: bool,
    title_tokenizer: TextTokenizer,
    abstract_tokenizer: TextTokenizer,
) -> Data:
    """
    Tokenizes the text data in the PyTorch Geometric Data object.

    Args:
        data (Data): The PyTorch Geometric Data object containing text data.
        is_train (bool): Whether the data is for training or not.
        title_tokenizer (TextTokenizer): Tokenizer for the titles.
        abstract_tokenizer (TextTokenizer): Tokenizer for the abstracts.

    Returns:
        Data: The updated Data object with tokenized text.
    """

    titles = list(data.x[:, 0])
    abstracts = list(data.x[:, 1])

    tokenized_titles = title_tokenizer(titles).convert_to_tensors()
    tokenized_abstracts = abstract_tokenizer(abstracts).convert_to_tensors()

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
