import argparse

from .script_args import (
    DataGenerationAC,
    TrainGraphModelAC,
)


class ArgumentsFactory:

    @staticmethod
    def get_args(type: str):

        ARGUMENT_CREATORS = dict(
            {
                "data_generation": DataGenerationAC,
                "train_graph_model": TrainGraphModelAC,
            }
        )

        if type not in ARGUMENT_CREATORS:
            raise ValueError(f"Invalid argument type: {type}")

        return ARGUMENT_CREATORS[type].create_parser().parse_args()
