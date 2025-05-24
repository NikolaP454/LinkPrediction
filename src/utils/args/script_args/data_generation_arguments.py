import argparse
from .base_argument import BaseArgumentCreator


class DataGenerationAC(BaseArgumentCreator):

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Data Generation Arguments")

        parser.add_argument(
            "--input_data_path",
            type=str,
            required=True,
            help="Path to the input data file",
        )

        parser.add_argument(
            "--experiment_path",
            type=str,
            required=True,
            help="Path to the experiment directory where results will be saved",
        )

        parser.add_argument(
            "--dataset",
            type=str,
            default="ogbn-arxiv",
            help="Name of the dataset to be used for data generation",
        )

        parser.add_argument(
            "--negative_edges_per_node",
            type=int,
            default=10,
            help="Number of negative edges to generate per node",
        )

        return parser
