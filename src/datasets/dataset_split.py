from tqdm import tqdm
from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data


def generate_split_masks(
    dataset: Data,
    train_idx: Tensor,
    val_idx: Tensor,
    test_idx: Tensor,
) -> Data:
    """
    Generates masks for the training, validation, and test sets.

    Args:
        dataset (Data): The PyTorch Geometric dataset object.
        train_idx (Tensor): The indices of the training set.
        val_idx (Tensor): The indices of the validation set.
        test_idx (Tensor): The indices of the test set.

    Returns:
        Data: A new Data object with the original data and the generated masks.
    """
    num_nodes = dataset.num_nodes

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return Data(
        x=dataset.x,
        edge_index=dataset.edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def generate_train_dataset(
    dataset: Data,
) -> Data:
    """
    Generates a training dataset with the specified indices.

    Args:
        dataset (Data): The PyTorch Geometric dataset object.

    Returns:
        Data: A new Data object with the original data and the generated training dataset.
    """

    if not hasattr(dataset, "train_mask"):
        raise ValueError("The dataset does not have a train_mask attribute.")

    train_mask = dataset.train_mask

    # Create a mapping from the original node indices to the new indices
    counter, node_idx_mapping = 0, dict()

    for og_idx, mask in enumerate(train_mask):
        if not mask:
            continue

        node_idx_mapping[og_idx] = counter
        counter += 1

    # Create a new edge index based on the mapping
    new_edge_index = []
    for edge in tqdm(dataset.edge_index.t(), desc="Generating training dataset"):
        if not train_mask[edge[0]]:
            continue
        if not train_mask[edge[1]]:
            continue

        new_edge_index.append(
            torch.tensor(
                [
                    node_idx_mapping[edge[0].item()],
                    node_idx_mapping[edge[1].item()],
                ]
            )
        )

    new_edge_index = torch.stack(new_edge_index).t()

    return Data(
        x=dataset.x[train_mask],
        edge_index=new_edge_index.contiguous(),
    )
