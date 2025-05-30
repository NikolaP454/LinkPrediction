import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryAUPRC
from torch_geometric.loader import LinkNeighborLoader

from .. import datasets


def evaluate_model(
    model: nn.Module,
    test_loader: LinkNeighborLoader,
    device: str = "cpu",
    results_path: str = None,
) -> None:
    """
    Evaluate the model on the validation set and save the results.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (LinkNeighborLoader): The data loader for testing.
        device (str): The device to use for evaluation ('cpu' or 'cuda').
        results_path (str, optional): Path to save the evaluation results. Defaults to None.
    """

    # Setup
    model.eval()
    model.to(device)
    print(f"Evaluating model on device: {model.device}")

    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    auprc = BinaryAUPRC().to(device)

    # Extract predictions and labels
    all_predictions = []
    all_labels = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        title_inputs = batch.x[:, 0]
        abstract_inputs = batch.x[:, 1]

        edge_index = batch["edge_index"].to(device)
        edge_label_index = batch["edge_label_index"].to(device)
        edge_label = batch["edge_label"].to(device)

        pred_label = model(
            title_inputs,
            abstract_inputs,
            edge_index,
            edge_label_index,
        ).squeeze()

        all_predictions.append(pred_label.detach().cpu())
        all_labels.append(edge_label.detach().cpu())

    # Calculate precision, recall and F1 score
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    precision.update(all_predictions, all_labels)
    recall.update(all_predictions, all_labels)
    auprc.update(all_predictions, all_labels)

    precision_value = precision.compute().item()
    recall_value = recall.compute().item()
    auprc_value = auprc.compute().item()

    f1_score = 0.0
    if precision_value + recall_value > 0:
        f1_score = (
            2 * (precision_value * recall_value) / (precision_value + recall_value)
        )

    # Output evaluation results
    print("Evaluation Results:")
    print(
        f"Precision: {precision_value:.4f}, Recall: {recall_value:.4f}, F1 Score: {f1_score:.4f}"
    )

    print(f"AUPRC: {auprc_value:.4f}")

    if results_path:
        results_file = os.path.join(results_path, "evaluation_results.csv")

        with open(results_file, "w") as f:
            f.write("Metric,Value\n")
            f.write(f"Precision,{precision_value:.4f}\n")
            f.write(f"Recall,{recall_value:.4f}\n")
            f.write(f"F1 Score,{f1_score:.4f}\n")
            f.write(f"AUPRC,{auprc_value:.4f}\n")

        print(f"Evaluation results saved to {results_file}")
