import os
import gzip
import pandas as pd

from tqdm import tqdm

import torch
from torch import Tensor
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data


def generate_dataset(name: str, root: str) -> None:
    """
    Generates a dataset for the given dataset name and saves it to the specified root directory.

    Args:
        name (str): The name of the dataset to generate.
        root (str): The root directory to save the dataset.
        
    Returns:
        dataset_pg (Data): The PyTorch Geometric dataset object.
        train_idx (Tensor): The indices of the training set.
        valid_idx (Tensor): The indices of the validation set.
        test_idx (Tensor): The indices of the test set.
    """

    # Load the graph data
    dataset = PygNodePropPredDataset(name=name, root=root)
    
    graph = dataset[0]
    
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )

    # Load the title and abstract data
    TITLE_ABS_PATH = os.path.join(root, "titleabs.tsv")

    titles_df = pd.read_csv(TITLE_ABS_PATH, sep="\t")
    titles_df.columns = ["id", "title", "abstract"]
    titles_df

    MAPPING_FILE_PATH = os.path.join(
        root, "ogbn_arxiv", "mapping", "nodeidx2paperid.csv.gz"
    )

    with gzip.open(MAPPING_FILE_PATH, "rt") as f:
        mapping_df = pd.read_csv(f)

    mapping_df.columns = ["node_id", "paper_id"]

    # Merge the dataframes
    merged_df = titles_df.merge(
        mapping_df, how="inner", left_on="id", right_on="paper_id", suffixes=("", "_y")
    )

    merged_df = merged_df[["node_id", "id", "title", "abstract"]]
    merged_df["node_id"] = merged_df["node_id"].astype(int)
    merged_df["id"] = merged_df["id"].astype(int)

    merged_df.sort_values(by="node_id", inplace=True)

    dataset_pg = Data(
        x=merged_df[["title", "abstract"]].values,
        edge_index=graph.edge_index,
    )

    return dataset_pg, train_idx, valid_idx, test_idx