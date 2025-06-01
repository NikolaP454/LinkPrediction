import os
import json

import torch
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from src import (
    datasets,
    utils,
    models,
)

if __name__ == "__main__":
    ARGUMENTS = utils.args.ArgumentsFactory.get_args("evaluate_prompt_model")

    # Locate the experiment paths
    EXPERIMENT_PATH = ARGUMENTS.experiment_path
    EXPERIMENT_DATA_PATH = os.path.join(EXPERIMENT_PATH, "data")
    EXPERIMENT_RUN_CONFIGS_PATH = os.path.join(EXPERIMENT_PATH, "run_configs")

    assert os.path.exists(
        EXPERIMENT_PATH
    ), f"Experiment path {EXPERIMENT_PATH} does not exist."
    assert os.path.exists(
        EXPERIMENT_DATA_PATH
    ), f"Experiment data path {EXPERIMENT_DATA_PATH} does not exist."
    assert os.path.exists(
        EXPERIMENT_RUN_CONFIGS_PATH
    ), f"Experiment run configs path {EXPERIMENT_RUN_CONFIGS_PATH} does not exist."

    EXPERIMENT_MODEL_PATH = os.path.join(EXPERIMENT_PATH, "prompt_models")
    MODEL_PATH = os.path.join(EXPERIMENT_MODEL_PATH, ARGUMENTS.model_name)

    assert os.path.exists(
        EXPERIMENT_MODEL_PATH
    ), f"Experiment model path {EXPERIMENT_MODEL_PATH} does not exist."
    assert os.path.exists(MODEL_PATH), f"Model path {MODEL_PATH} does not exist."

    # -- Evaluation Paths
    EVALUATION_RESULTS_PATH = os.path.join(EXPERIMENT_PATH, "evaluation_results")
    os.makedirs(EVALUATION_RESULTS_PATH, exist_ok=True)

    CURRENT_EVALUATION_RESULTS_PATH = os.path.join(
        EVALUATION_RESULTS_PATH,
        ARGUMENTS.model_name,
        f"epoch_{ARGUMENTS.model_iteration}",
    )
    os.makedirs(CURRENT_EVALUATION_RESULTS_PATH, exist_ok=True)

    # Model Related Arguments
    MODEL_ITERATION = ARGUMENTS.model_iteration

    # Save the run configurations
    with open(
        os.path.join(
            CURRENT_EVALUATION_RESULTS_PATH,
            f"evaluation_{ARGUMENTS.model_iteration}_args.json",
        ),
        "w",
    ) as f:
        json.dump(ARGUMENTS.__dict__, f, indent=4)

    # Other configurations
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Import training configurations
    train_configs_path = os.path.join(
        EXPERIMENT_RUN_CONFIGS_PATH, f"train_prompt_{ARGUMENTS.model_name}_args.json"
    )
    train_configs = json.load(open(train_configs_path, "r"))

    MODEL_MAX_LENGTH = train_configs["model_max_length"]

    LOADER_NEIGHBORHOOD_SIZE = train_configs["loader_neighborhood_size"]
    LOADER_BATCH_SIZE = train_configs["loader_batch_size"]
    LOADER_DEPTH = train_configs["loader_depth"]

    ADD_PAPER_RELATIONS = train_configs["add_paper_relations"]
    MAX_PAPERS_PER_RELATION = train_configs["max_papers_per_relation"]

    # Load the dataset
    test_data = torch.load(os.path.join(EXPERIMENT_DATA_PATH, "test_data.pt"))
    test_arxiv_dataset = datasets.ArxivDataset(test_data)

    test_loader = LinkNeighborLoader(
        data=test_arxiv_dataset.get_data(),
        num_neighbors=[LOADER_NEIGHBORHOOD_SIZE] * LOADER_DEPTH,
        neg_sampling_ratio=0,
        batch_size=LOADER_BATCH_SIZE,
        shuffle=False,
        edge_label_index=test_data.edge_index,
        edge_label=test_data.edge_label,
    )

    # Load the model
    model = models.prompt_models.PromptLinkPredictionModel(
        max_length=MODEL_MAX_LENGTH,
    )

    model.load_pretrained(
        os.path.join(MODEL_PATH, f"model_{MODEL_ITERATION}.pt"), device=DEVICE
    )

    # Evaluate the model
    utils.evaluation.evaluate_prompt_model(
        model=model,
        test_dataset=test_arxiv_dataset,
        test_loader=test_loader,
        add_paper_relations=ADD_PAPER_RELATIONS,
        max_papers_per_relation=MAX_PAPERS_PER_RELATION,
        device=DEVICE,
        results_path=CURRENT_EVALUATION_RESULTS_PATH,
    )
