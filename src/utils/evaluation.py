import os, sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryAUPRC
from torch_geometric.loader import LinkNeighborLoader

from .. import datasets
from . import prompting


def get_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """
    Calculate precision, recall, F1 score, and AUPRC.

    Args:
        predictions (torch.Tensor): Model predictions.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        tuple: Precision, recall, F1 score, and AUPRC values.
    """
    precision = BinaryPrecision()
    recall = BinaryRecall()
    auprc = BinaryAUPRC()

    precision.update(predictions, labels)
    recall.update(predictions, labels)
    auprc.update(predictions, labels)

    precision_value = precision.compute().item()
    recall_value = recall.compute().item()

    f1_score = 0.0
    if precision_value + recall_value > 0:
        f1_score = (
            2 * (precision_value * recall_value) / (precision_value + recall_value)
        )

    auprc_value = auprc.compute().item()

    results = {
        "precision": precision_value,
        "recall": recall_value,
        "f1_score": f1_score,
        "auprc": auprc_value,
    }

    return results


def print_and_save_results(
    results: dict,
    results_path: str = None,
) -> None:
    """
    Print and save evaluation results.

    Args:
        results (dict): Dictionary containing evaluation metrics.
        results_path (str, optional): Path to save the evaluation results. Defaults to None.
    """

    precision_value = results["precision"]
    recall_value = results["recall"]
    f1_score = results["f1_score"]
    auprc_value = results["auprc"]

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

        print(f"Evaluation results saved to {results_file}", file=sys.stderr)


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

    results = get_metrics(all_predictions, all_labels)
    print_and_save_results(results, results_path)


def evaluate_prompt_model(
    model: nn.Module,
    test_dataset: datasets.ArxivDataset,
    test_loader: LinkNeighborLoader,
    add_paper_relations: bool = False,
    max_papers_per_relation: int = 2,
    device: str = "cpu",
    results_path: str = None,
) -> None:
    """
    Evaluate the prompt model on the test dataset and save the results.

    Args:
        model (nn.Module): The prompt model to evaluate.
        test_dataset (datasets.ArxivDataset): The test dataset.
        test_loader (LinkNeighborLoader): The data loader for testing.
        add_paper_relations (bool): Whether to add paper relations in the evaluation.
        max_papers_per_relation (int): Maximum number of papers per relation to consider.
        device (str): The device to use for evaluation ('cpu' or 'cuda').
        results_path (str, optional): Path to save the evaluation results. Defaults to None.
    """

    # Setup
    model.eval()
    model.to(device)
    print(f"Evaluating model on device: {model.device}")

    # Extract predictions and labels
    all_predictions = []
    all_labels = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        global_n_ids = batch["x"].tolist()

        edge_label_index = batch["edge_label_index"].to(device)

        prompts = []

        for i in range(edge_label_index.size(1)):
            src, dst = int(edge_label_index[0, i]), int(edge_label_index[1, i])

            source_id = int(global_n_ids[src])
            destination_id = int(global_n_ids[dst])

            source_title = test_dataset.get_title(source_id)
            destination_title = test_dataset.get_title(destination_id)

            base_prompt = prompting.generate_base_prompt(
                source_title, destination_title
            )

            prompt = base_prompt

            if add_paper_relations:
                source_cites = test_dataset.get_cites(source_id, ignore=destination_id)
                source_is_cited_by = test_dataset.get_is_cited_by(
                    source_id, ignore=destination_id
                )

                destination_cites = test_dataset.get_cites(
                    destination_id, ignore=source_id
                )
                destination_is_cited_by = test_dataset.get_is_cited_by(
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

        edge_label = batch["edge_label"].to(device)
        pred_label = model(prompts).squeeze()

        all_predictions.append(pred_label.detach().cpu())
        all_labels.append(edge_label.detach().cpu())

    # Calculate precision, recall and F1 score
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    results = get_metrics(all_predictions, all_labels)
    print_and_save_results(results, results_path)
