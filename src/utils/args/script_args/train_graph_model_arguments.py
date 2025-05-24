import os
import argparse
from .base_argument import BaseArgumentCreator


class TrainGraphModelAC(BaseArgumentCreator):

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
            "--hidden_channel_size",
            type=int,
            default=64,
            help="Size of the hidden channels in the graph model",
        )

        parser.add_argument(
            "--output_channel_size",
            type=int,
            default=128,
            help="Size of the output channels in the graph model",
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
            default=10,
            help="Size of the neighborhood for the data loader",
        )

        parser.add_argument(
            "--loader_batch_size",
            type=int,
            default=128,
            help="Batch size for the data loader",
        )

        parser.add_argument(
            "--epochs",
            type=int,
            default=1,
            help="Number of epochs to train the model",
        )

        return parser
