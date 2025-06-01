import argparse

from .script_args import (
    DataGenerationAC,
    TrainGraphModelAC,
    TrainPromptModelAC,
    EvaluateGraphModelAC,
    EvaluatePromptModelAC,
)


class ArgumentsFactory:

    @staticmethod
    def get_args(type: str):

        ARGUMENT_CREATORS = dict(
            {
                "data_generation": DataGenerationAC,
                "train_graph_model": TrainGraphModelAC,
                "train_prompt_model": TrainPromptModelAC,
                "evaluate_graph_model": EvaluateGraphModelAC,
                "evaluate_prompt_model": EvaluatePromptModelAC,
            }
        )

        if type not in ARGUMENT_CREATORS:
            raise ValueError(f"Invalid argument type: {type}")

        return ARGUMENT_CREATORS[type].create_parser().parse_args()
