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
    ARGUMENTS = utils.args.ArgumentsFactory.get_args("evaluate_graph_model")

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

    EXPERIMENT_MODEL_PATH = os.path.join(EXPERIMENT_PATH, "graph_models")
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

    LOADER_NEIGHBORHOOD_SIZE = ARGUMENTS.loader_neighborhood_size
    LOADER_BATCH_SIZE = ARGUMENTS.loader_batch_size

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
        EXPERIMENT_RUN_CONFIGS_PATH, f"train_graph_{ARGUMENTS.model_name}_args.json"
    )
    train_configs = json.load(open(train_configs_path, "r"))

    if LOADER_NEIGHBORHOOD_SIZE == -1:
        LOADER_NEIGHBORHOOD_SIZE = train_configs["loader_neighborhood_size"]

    if LOADER_BATCH_SIZE == -1:
        LOADER_BATCH_SIZE = train_configs["loader_batch_size"]

    # Load the dataset
    data = torch.load(os.path.join(EXPERIMENT_DATA_PATH, "test_data_tokenized.pt"))
    test_dataset = datasets.ArxivDataset(data)

    train_loader = LinkNeighborLoader(
        test_dataset.get_data(),
        num_neighbors=[LOADER_NEIGHBORHOOD_SIZE] * 2,
        neg_sampling_ratio=1,
        batch_size=LOADER_BATCH_SIZE,
        shuffle=True,
    )

    # Load the model
    model = models.graph_text_models.SageConvModel(
        hidden_channels=train_configs["hidden_channel_size"],
        out_channels=train_configs["output_channel_size"],
        use_titles=not train_configs["ignore_titles"],
        use_abstracts=not train_configs["ignore_abstracts"],
        reduced_dim_titles=train_configs["reduced_dim_titles"],
        reduced_dim_abstracts=train_configs["reduced_dim_abstracts"],
    )

    model.load_pretrained(
        os.path.join(MODEL_PATH, f"model_{MODEL_ITERATION}.pt"), device=DEVICE
    )
