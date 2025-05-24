import argparse


class BaseArgumentCreator:

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        raise NotImplementedError("This method should be overridden by subclasses")
