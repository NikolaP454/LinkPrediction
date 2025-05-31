import os, sys
import json

import torch
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling

from src import (
    datasets,
    utils,
    models,
)


if __name__ == "__main__":
    ARGUMENTS = utils.args.ArgumentsFactory.get_args("train_prompt_model")

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

    os.makedirs(EXPERIMENT_MODEL_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

    CONTINUE_FROM_EPOCH = ARGUMENTS.continue_from_epoch

    # Model Related Arguments
    MODEL_NAME = ARGUMENTS.model_name
    MODEL_MAX_LENGTH = ARGUMENTS.model_max_length

    USE_TITLES = not ARGUMENTS.ignore_titles
    USE_ABSTRACTS = not ARGUMENTS.ignore_abstracts

    ADD_PAPER_RELATIONS = ARGUMENTS.add_paper_relations
    MAX_PAPERS_PER_RELATION = ARGUMENTS.max_papers_per_relation

    INITIAL_LR = ARGUMENTS.initial_lr
    WEIGHT_DECAY = ARGUMENTS.weight_decay
    EPOCHS = ARGUMENTS.epochs

    LOADER_NEIGHBORHOOD_SIZE = ARGUMENTS.loader_neighborhood_size
    LOADER_BATCH_SIZE = ARGUMENTS.loader_batch_size
    LOADER_DEPTH = ARGUMENTS.loader_depth

    # Save the run configurations
    with open(
        os.path.join(
            EXPERIMENT_RUN_CONFIGS_PATH, f"train_prompt_{MODEL_NAME}_args.json"
        ),
        "w",
    ) as f:
        json.dump(ARGUMENTS.__dict__, f, indent=4)

    # Other configurations
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    train_data = torch.load(os.path.join(EXPERIMENT_DATA_PATH, "train_data.pt"))

    train_arxiv_data = datasets.ArxivDataset(train_data)

    train_loader = LinkNeighborLoader(
        data=train_arxiv_data.get_data(),
        num_neighbors=[LOADER_NEIGHBORHOOD_SIZE] * LOADER_DEPTH,
        neg_sampling=NegativeSampling(mode="triplet", amount=1),
        batch_size=LOADER_BATCH_SIZE,
        shuffle=True,
    )

    # Initialize the model
    model = models.prompt_models.PromptLinkPredictionModel(
        max_length=MODEL_MAX_LENGTH,
    )

    optimizer = Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)

    # Train the model
    utils.training.train_prompt_model(
        model=model,
        optimizer=optimizer,
        dataset=train_arxiv_data,
        train_loader=train_loader,
        add_relations=ADD_PAPER_RELATIONS,
        max_papers_per_relation=MAX_PAPERS_PER_RELATION,
        device=DEVICE,
        epochs=EPOCHS,
        model_path=MODEL_PATH,
        continue_from_epoch=CONTINUE_FROM_EPOCH,
    )

    print(f"Model trained and saved at {MODEL_PATH}", file=sys.stderr)
