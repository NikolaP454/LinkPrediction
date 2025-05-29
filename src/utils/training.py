import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch_geometric.loader import LinkNeighborLoader

from .. import datasets


def train_model(
    model: nn.Module,
    train_loader: LinkNeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    epochs: int = 1,
    model_path: str = None,
    continue_from_epoch: int = 0,
) -> None:
    """
    Train the model on the dataset.

    Args:
        model (nn.Module): The model to train.
        train_loader (LinkNeighborLoader): The data loader for training.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (str): The device to use for training ('cpu' or 'cuda').
        epochs (int): The number of epochs to train for.
        model_path (str, optional): Path to save the model after training. Defaults to None.
        continue_from_epoch (int): Epoch to continue training from. Defaults to 0.
    """

    # Sanity checks
    assert (
        continue_from_epoch < epochs
    ), "Can't continue from an epoch greater than or equal to total epochs."

    print(f"Training on device: {device}")

    # Model Setup
    if continue_from_epoch > 0 and model_path:
        model.load_pretrained(
            os.path.join(model_path, f"model_{continue_from_epoch}.pt"), device
        )

    model.train()
    model.to(device)

    RANGE = (
        range(continue_from_epoch, epochs) if continue_from_epoch > 0 else range(epochs)
    )

    for epoch in RANGE:
        total_loss = 0.0

        # Process each batch in the training loader
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            title_inputs = batch.x[:, 0].to(device)
            abstract_inputs = batch.x[:, 1].to(device)

            edge_index = batch["edge_index"].to(device)
            edge_label_index = batch["edge_label_index"].to(device)
            edge_label = batch["edge_label"].to(device)

            pred_label = model(
                title_inputs,
                abstract_inputs,
                edge_index,
                edge_label_index,
            ).squeeze()

            loss = nn.BCEWithLogitsLoss()(pred_label, edge_label.float())
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Logging
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        if model_path:
            torch.save(
                model.state_dict(), os.path.join(model_path, f"model_{epoch}.pt")
            )
