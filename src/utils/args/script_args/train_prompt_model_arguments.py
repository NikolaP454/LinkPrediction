import os
import argparse
from .base_argument import BaseArgumentCreator


class TrainPromptModelAC(BaseArgumentCreator):

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Train Graph Model Arguments")

        parser.add_argument(
            "--experiment_path",
            type=str,
            required=True,
            help="Path to the experiment directory where results will be saved",
        )

        parser.add_argument(
            "--model_name",
            type=str,
            required=True,
            help="Name of the graph model to be trained",
        )

        parser.add_argument(
            "--model_max_length",
            type=int,
            default=128,
            help="Maximum length of the model input sequences",
        )

        parser.add_argument(
            "--initial_lr",
            type=float,
            default=0.01,
            help="Initial learning rate for the optimizer",
        )

        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.0001,
            help="Weight decay for the optimizer",
        )

        parser.add_argument(
            "--loader_neighborhood_size",
            type=int,
            default=3,
            help="Size of the neighborhood for the data loader",
        )

        parser.add_argument(
            "--loader_batch_size",
            type=int,
            default=16,
            help="Batch size for the data loader",
        )

        parser.add_argument(
            "--loader_depth",
            type=int,
            default=1,
            help="Depth of the neighborhood for the data loader",
        )

        parser.add_argument(
            "--epochs",
            type=int,
            default=1,
            help="Number of epochs to train the model",
        )

        parser.add_argument(
            "--ignore_titles",
            action="store_true",
            default=False,
            help="Ignore titles in the graph model training",
        )

        parser.add_argument(
            "--ignore_abstracts",
            action="store_true",
            default=False,
            help="Ignore abstracts in the graph model training",
        )

        parser.add_argument(
            "--continue_from_epoch",
            type=int,
            default=0,
            help="Epoch to continue training from, 0 means start from scratch",
        )

        parser.add_argument(
            "--add_paper_relations",
            action="store_true",
            default=False,
            help="Whether to add paper relations during training",
        )

        parser.add_argument(
            "--max_papers_per_relation",
            type=int,
            default=2,
            help="Maximum number of papers to include in each relation",
        )

        return parser
