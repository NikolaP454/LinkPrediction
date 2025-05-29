import os
import argparse
from .base_argument import BaseArgumentCreator


class EvaluateGraphModelAC(BaseArgumentCreator):

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Evaluating Graph Model Arguments")

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
            "--model_iteration",
            type=int,
            default=0,
            help="Iteration of the model to be evaluated",
        )

        parser.add_argument(
            "--loader_neighborhood_size",
            type=int,
            default=-1,
            help="Size of the neighborhood for the data loader (or -1 for training default)",
        )

        parser.add_argument(
            "--loader_batch_size",
            type=int,
            default=-1,
            help="Batch size for the data loader (or -1 for training default)",
        )

        return parser
