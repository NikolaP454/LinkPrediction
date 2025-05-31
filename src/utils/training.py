import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torcheval.metrics import BinaryPrecision, BinaryRecall
from torch_geometric.loader import LinkNeighborLoader

from .. import datasets
from . import prompting


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

    loss_f = nn.BCELoss()

    for epoch in RANGE:
        total_loss = 0.0

        # Process each batch in the training loader
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            title_inputs = batch.x[:, 0].to(device)
            abstract_inputs = batch.x[:, 1].to(device)

            edge_index = batch["edge_index"].to(device)
            # edge_label_index = batch["edge_label_index"].to(device)
            # edge_label = batch["edge_label"].to(device)

            src_index = batch["src_index"].to(device)
            dst_index = batch["dst_pos_index"].to(device)
            dst_neg_index = batch["dst_neg_index"].to(device)

            postive_edge_index = torch.stack([src_index, dst_index], dim=0)
            negative_edge_index = torch.stack([src_index, dst_neg_index], dim=0)

            edge_label_index = torch.cat(
                [postive_edge_index, negative_edge_index], dim=1
            )
            edge_label = torch.cat(
                [
                    torch.ones(postive_edge_index.shape[1], device=device),
                    torch.zeros(negative_edge_index.shape[1], device=device),
                ],
                dim=0,
            )

            pred_label = model(
                title_inputs,
                abstract_inputs,
                edge_index,
                edge_label_index,
            ).squeeze()

            loss = loss_f(pred_label, edge_label.float())
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        if model_path:
            torch.save(
                model.state_dict(), os.path.join(model_path, f"model_{epoch}.pt")
            )


def train_prompt_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: datasets.ArxivDataset,
    train_loader: LinkNeighborLoader,
    add_relations: bool = False,
    max_papers_per_relation: int = 2,
    device: str = "cpu",
    epochs: int = 1,
    model_path: str = None,
    continue_from_epoch: int = 0,
):
    """
    Train the prompt model on the dataset.

    Args:
        model (nn.Module): The prompt model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        dataset (datasets.ArxivDataset): The dataset to train on.
        train_loader (LinkNeighborLoader): The data loader for training.
        add_relations (bool): Whether to add paper relations during training.
        max_papers_per_relation (int): Maximum number of papers per relation to consider.
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
            os.path.join(model_path, f"model_{continue_from_epoch - 1}.pt"), device
        )

    model.train()
    model.to(device)

    RANGE = (
        range(continue_from_epoch, epochs) if continue_from_epoch > 0 else range(epochs)
    )

    loss_f = nn.BCELoss()

    for epoch in RANGE:
        total_loss = 0.0

        # Process each batch in the training loader
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            # Extracting titles and abstracts from the dataset
            titles = [dataset.get_title(i) for i in batch.x[:, 0].tolist()]
            abstracts = [dataset.get_abstract(i) for i in batch.x[:, 1].tolist()]

            # Extracting node IDs and indices
            src_index = batch["src_index"].to(device)
            dst_pos_index = batch["dst_pos_index"].to(device)
            dst_neg_index = batch["dst_neg_index"].to(device)
            dst_index = torch.cat([dst_pos_index, dst_neg_index], dim=0)

            source_ids = batch["x"][src_index.tolist()].to(device)
            destination_ids = batch["x"][dst_index.tolist()].to(device)

            # Generating prompts for the model
            prompts = []

            for source_id, src, destination_id, dst in zip(
                source_ids, src_index, destination_ids, dst_index
            ):

                source_title, _ = titles[src], abstracts[src]
                destination_title, _ = titles[dst], abstracts[dst]

                base_prompt = prompting.generate_base_prompt(
                    source_title, destination_title
                )

                prompt = base_prompt

                if add_relations:
                    source_cites = dataset.get_cites(source_id, ignore=destination_id)
                    source_is_cited_by = dataset.get_cites(
                        source_id, ignore=destination_id
                    )

                    destination_cites = dataset.get_cites(
                        destination_id, ignore=source_id
                    )
                    destination_is_cited_by = dataset.get_cites(
                        destination_id, ignore=source_id
                    )

                    source_added_prompt = prompting.add_paper_relations(
                        is_source=True,
                        cites=source_cites,
                        is_cited_by=source_is_cited_by,
                        max_papers=max_papers_per_relation,
                        source_prompt=base_prompt,
                    )

                    prompt = prompting.add_paper_relations(
                        is_source=False,
                        cites=destination_cites,
                        is_cited_by=destination_is_cited_by,
                        max_papers=max_papers_per_relation,
                        source_prompt=source_added_prompt,
                    )

                prompts.append(prompt)

            # Genrating labels for the edges
            pred_label = model(prompts).squeeze()
            edge_label = torch.cat(
                [
                    torch.ones(len(dst_pos_index), device=device),
                    torch.zeros(len(dst_neg_index), device=device),
                ],
                dim=0,
            )

            # Backpropagation
            loss = loss_f(pred_label, edge_label.float())
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        if model_path:
            torch.save(
                model.state_dict(), os.path.join(model_path, f"model_{epoch}.pt")
            )
