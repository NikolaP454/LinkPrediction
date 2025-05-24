import random
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


def generate_test_dataset(
    dataset: Data,
    negative_edges_per_node: int = 10,
) -> Data:
    """
    Generates a test dataset with the specified indices.

    Args:
        dataset (Data): The PyTorch Geometric dataset object.
        negative_edges_per_node (int): The number of negative edges to generate per node.

    Returns:
        Data: A new Data object with the original data and the generated test dataset.
    """

    if not hasattr(dataset, "test_mask"):
        raise ValueError("The dataset does not have a test_mask attribute.")

    test_mask = dataset.test_mask

    # Create a new edge index based on the mapping
    new_edge_index = []
    for edge in tqdm(dataset.edge_index.t(), desc="Generating test dataset"):
        if not test_mask[edge[0]]:
            continue

        new_edge_index.append(
            torch.tensor(
                [
                    edge[0].item(),
                    edge[1].item(),
                ]
            )
        )

    new_edge_index = torch.stack(new_edge_index).t()
    new_edge_label = torch.ones(new_edge_index.size(1), dtype=torch.int)

    new_nodes = set(new_edge_index[0].tolist()).union(set(new_edge_index[1].tolist()))
    new_nodes = sorted(new_nodes)

    testing_nodes_in_new = set(new_edge_index[0].tolist())
    testing_nodes_existing = dict()

    # Generate adjacency list for existing edges
    for edge in tqdm(
        new_edge_index.t(), desc="Generating test dataset - Existing Edges"
    ):
        source, target = edge[0].item(), edge[1].item()

        if source not in testing_nodes_existing:
            testing_nodes_existing[source] = set()

        testing_nodes_existing[source].add(target)

    # Generate negative edges for the test dataset
    negative_edge_index = []

    for node in tqdm(
        testing_nodes_in_new, desc="Generating test dataset - Negative Edges"
    ):
        node_counter = 0

        if node not in new_nodes:
            raise ValueError(f"Node {node} not found in new nodes.")

        node_neighbors = testing_nodes_existing.get(node, set())

        for target in random.sample(
            new_nodes,
            min(negative_edges_per_node + len(node_neighbors) + 1, len(new_nodes)),
        ):
            if node_counter >= negative_edges_per_node:
                break

            if target == node:
                continue

            if target not in node_neighbors:
                negative_edge_index.append(torch.tensor([node, target]))
                node_counter += 1

    negative_edge_index = torch.stack(negative_edge_index).t()

    new_edge_index = torch.cat([new_edge_index, negative_edge_index], dim=1)
    new_edge_label = torch.cat(
        [new_edge_label, torch.zeros(negative_edge_index.size(1), dtype=torch.int)],
        dim=0,
    )

    # Create a mapping from the original node indices to the new indices
    node_idx_mapping = dict()

    for new_idx, og_idx in enumerate(new_nodes):
        node_idx_mapping[og_idx] = new_idx

    new_edge_index.apply_(lambda x: node_idx_mapping[x])

    return Data(
        x=dataset.x[new_nodes],
        edge_index=new_edge_index.contiguous(),
        edge_label=new_edge_label.contiguous(),
    )
