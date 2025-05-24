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
    ARGUMENTS = utils.args.ArgumentsFactory.get_args("train_graph_model")

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
    os.makedirs(EXPERIMENT_MODEL_PATH, exist_ok=True)

    # Model Related Arguments
    MODEL_NAME = ARGUMENTS.model_name
    HIDDEN_CHANNEL_SIZE = ARGUMENTS.hidden_channel_size
    OUTPUT_CHANNEL_SIZE = ARGUMENTS.output_channel_size

    INITIAL_LR = ARGUMENTS.initial_lr
    WEIGHT_DECAY = ARGUMENTS.weight_decay
    EPOCHS = ARGUMENTS.epochs

    LOADER_NEIGHBORHOOD_SIZE = ARGUMENTS.loader_neighborhood_size
    LOADER_BATCH_SIZE = ARGUMENTS.loader_batch_size

    # Save the run configurations
    with open(
        os.path.join(EXPERIMENT_RUN_CONFIGS_PATH, "train_graph_model_args.json"), "w"
    ) as f:
        json.dump(ARGUMENTS.__dict__, f, indent=4)

    # Other configurations
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    data = torch.load(os.path.join(EXPERIMENT_DATA_PATH, "train_data.pt"))
    train_dataset = datasets.ArxivDataset(data)

    train_loader = LinkNeighborLoader(
        train_dataset.get_data(),
        num_neighbors=[LOADER_NEIGHBORHOOD_SIZE] * 2,
        neg_sampling_ratio=1,
        batch_size=LOADER_BATCH_SIZE,
        shuffle=True,
    )

    # Initialize the model
    model = models.graph_text_models.SageConvModel(
        in_channels=0,
        hidden_channels=HIDDEN_CHANNEL_SIZE,
        out_channels=OUTPUT_CHANNEL_SIZE,
    )

    optimizer = Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)

    # Train the model
    utils.training.train_model(
        model=model,
        optimizer=optimizer,
        dataset=train_dataset,
        train_loader=train_loader,
        device=DEVICE,
        epochs=EPOCHS,
    )

    # Save the model
    model_save_path = os.path.join(EXPERIMENT_MODEL_PATH, f"{MODEL_NAME}.pt")
    torch.save(model.state_dict(), model_save_path)
