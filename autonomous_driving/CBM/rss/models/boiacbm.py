# CBM model for BOIA
from argparse import ArgumentParser
from utils.args import add_management_args, add_experiment_args
from models.utils.oia_cbm import OIACBM


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class BoiaCBM(OIACBM):
    """CBM MODEL FOR BOIA: the OIA CBM (models/utils/oia_cbm.py) on BDD-OIA"""

    NAME = "boiacbm"

    def __init__(
        self,
        encoder,
        n_images=1,
        c_split=(),
        args=None,
        model_dict=None,
        n_facts=21,
        nr_classes=4,
    ):
        """Initialize method

        Args:
            self: instance
            encoder (nn.Module): encoder
            n_images (int, default=1): number of images
            c_split: concept splits
            args: command line arguments
            model_dict (default=None): model dictionary
            n_facts (int, default=21): number of concepts
            nr_classes (int, nr_classes): number of classes for the multiclass classification problem

        Returns:
            None: This function does not return a value.
        """
        super(BoiaCBM, self).__init__(
            encoder,
            n_images=n_images,
            c_split=c_split,
            args=args,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )
