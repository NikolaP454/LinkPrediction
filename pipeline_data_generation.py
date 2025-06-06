import os, sys
import json
import torch

from src import (
    datasets,
    utils,
)


if __name__ == "__main__":
    ARGUMENTS = utils.args.ArgumentsFactory.get_args("data_generation")

    # Ensure the input data path exists
    INPUT_DATA_PATH = ARGUMENTS.input_data_path
    assert os.path.exists(
        INPUT_DATA_PATH
    ), f"Input data path {INPUT_DATA_PATH} does not exist."

    # Create directories for the experiment
    EXPERIMENT_PATH = ARGUMENTS.experiment_path
    EXPERIMENT_DATA_PATH = os.path.join(EXPERIMENT_PATH, "data")
    EXPERIMENT_RUN_CONFIGS_PATH = os.path.join(EXPERIMENT_PATH, "run_configs")

    os.makedirs(EXPERIMENT_PATH, exist_ok=True)
    os.makedirs(EXPERIMENT_DATA_PATH, exist_ok=True)
    os.makedirs(EXPERIMENT_RUN_CONFIGS_PATH, exist_ok=True)

    # Save the arguments to a file
    with open(
        os.path.join(EXPERIMENT_RUN_CONFIGS_PATH, "data_generation_args.json"), "w"
    ) as f:
        json.dump(ARGUMENTS.__dict__, f, indent=4)

    # Import the dataset
    dataset = ARGUMENTS.dataset
    NEGATIVE_EDGES_PER_NODE = ARGUMENTS.negative_edges_per_node

    data, train_idx, val_idx, test_idx = datasets.dataset_setup.generate_dataset(
        name=dataset,
        root=INPUT_DATA_PATH,
    )

    data = datasets.dataset_split.generate_split_masks(
        data,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    # Generate the train and test data
    train_data = datasets.dataset_split.generate_train_dataset(data)
    test_data = datasets.dataset_split.generate_test_dataset(
        data, negative_edges_per_node=NEGATIVE_EDGES_PER_NODE
    )

    # Tokenize the train and test data
    SHOULD_TOKENIZE = not ARGUMENTS.disable_tokenization

    if SHOULD_TOKENIZE:
        title_tokenizer = utils.tokenization.TextTokenizer()
        abstract_tokenizer = utils.tokenization.TextTokenizer()

        train_data = utils.tokenization.tokenize_data(
            train_data,
            is_train=True,
            title_tokenizer=title_tokenizer,
            abstract_tokenizer=abstract_tokenizer,
        )
        test_data = utils.tokenization.tokenize_data(
            test_data,
            is_train=False,
            title_tokenizer=title_tokenizer,
            abstract_tokenizer=abstract_tokenizer,
        )

    # Save the train and test data
    torch.save(
        train_data,
        os.path.join(
            EXPERIMENT_DATA_PATH,
            f"train_data{'_tokenized' if SHOULD_TOKENIZE else ''}.pt",
        ),
    )

    torch.save(
        test_data,
        os.path.join(
            EXPERIMENT_DATA_PATH,
            f"test_data{'_tokenized' if SHOULD_TOKENIZE else ''}.pt",
        ),
    )

    print("Data generation completed successfully.", file=sys.stderr)
    print(f"Generated train data with shape: {train_data.x.shape}", file=sys.stderr)
