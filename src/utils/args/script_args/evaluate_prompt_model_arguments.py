import os
import argparse
from .base_argument import BaseArgumentCreator


class EvaluatePromptModelAC(BaseArgumentCreator):

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Evaluating Prompt Model Arguments"
        )

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

        return parser
