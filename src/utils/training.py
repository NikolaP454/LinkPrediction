import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch_geometric.loader import LinkNeighborLoader

from .. import datasets


def train_model(
    model: nn.Module,
    dataset: datasets.ArxivDataset,
    train_loader: LinkNeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    epochs: int = 1,
) -> None:
    """
    Train the model on the dataset.

    Args:
        model (nn.Module): The model to train.
        dataset (datasets.ArxivDataset): The dataset to train on.
        train_loader (LinkNeighborLoader): The data loader for training.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (str): The device to use for training ('cpu' or 'cuda').
        epochs (int): The number of epochs to train for.
    """

    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            titles = [dataset.titles[int(i.item())] for i in batch["x"]]
            abstracts = [dataset.abstracts[int(i.item())] for i in batch["x"]]

            edge_index = batch["edge_index"].to(device)
            edge_label_index = batch["edge_label_index"].to(device)
            edge_label = batch["edge_label"].to(device)

            pred_label = model(
                titles,
                abstracts,
                edge_index,
                edge_label_index,
            ).squeeze()

            loss = nn.BCEWithLogitsLoss()(pred_label, edge_label.float())
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
