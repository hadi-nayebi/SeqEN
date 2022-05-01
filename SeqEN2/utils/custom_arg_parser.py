"""Define CustomArgParser and subclasses, for customized argument parsing."""
# TODO: Create annotated tag for version v0.0.2 (The Git way.)
# TODO: Replace __version__ constant with annotated tag. (The Git way.)
# TODO: Add --verbose flag ("-d" seems to be pre-empted, so "-v")
# TODO: Review the code itself

from argparse import ArgumentParser
from typing import Any, Dict

HelpValuePair = Dict[str, Any]


class CustomArgParser(ArgumentParser):
    """Add help_value_pairs method to ArgumentParser object."""

    def help_value_pairs(self) -> HelpValuePair:
        """Create and return dict of {option_help_message: option_value}."""
        parsed_args = self.parse_args()
        help_value_pair_dict = {}
        for value in self.__dict__["_option_string_actions"].values():
            if value.dest in parsed_args.__dict__.keys():
                help_value_pair_dict[value.help] = parsed_args.__dict__[value.dest]
        return help_value_pair_dict


class DefaultParser(object):
    """DefaultParser is the basic parser class. More specialized arg parsers will inherit from this class."""

    def __init__(self, desc: str) -> None:
        """Define instance variables, collect arguments."""
        self.parser = CustomArgParser(description=desc)
        self._initialize()

    def _initialize(self) -> None:
        """A virtual method."""

    def parsed(self) -> HelpValuePair:
        """Return description-value pair dictionary."""
        return self.parser.help_value_pairs()


class TrainSessionArgParser(DefaultParser):
    """Set description, options, and flags for TrainSession."""

    def __init__(self) -> None:
        """Define instance variables, collect arguments."""
        super().__init__("Train a protein sequence autoencoder")

    def _initialize(self) -> None:
        self.parser.add_argument("-n", "--model", type=str, help="Model Name", required=True)
        # dataset0: seq:act_p, dataset1: seq:ss, dataset2: seq:act_p:ss
        self.parser.add_argument("-dcl", "--dataset_cl", type=str, help="Dataset_cl", default="")
        self.parser.add_argument("-dss", "--dataset_ss", type=str, help="Dataset_ss", default="")
        self.parser.add_argument(
            "-dclss", "--dataset_clss", type=str, help="Dataset_clss", default=""
        )
        # architecture blueprint
        self.parser.add_argument("-a", "--arch", type=str, help="Arch", required=True)
        # madel hyper params, d1: amino acids vec, dn: embedding, w: sliding window
        self.parser.add_argument("-d1", "--d1", type=int, help="D1", default=8)
        self.parser.add_argument("-dn", "--dn", type=int, help="Dn", default=10)
        self.parser.add_argument("-w", "--w", type=int, help="W", default=20)
        self.parser.add_argument("-e", "--epochs", type=int, help="Epochs", default=25)
        self.parser.add_argument("-trb", "--train_batch", type=int, help="Train Batch", default=128)
        self.parser.add_argument("-teb", "--test_batch", type=int, help="Test Batch", default=1)
        self.parser.add_argument(
            "-ti", "--test_interval", type=int, help="Test Interval", default=100
        )
        self.parser.add_argument("-no", "--noise", type=float, help="Input Noise", default=0.0)
        self.parser.add_argument(
            "-mvid", "--model_version_id", type=str, help="Model Version ID", default=""
        )
        self.parser.add_argument(
            "-ts", "--training_settings", type=str, help="Training Settings", default=None
        )
        self.parser.add_argument(
            "-mts",
            "--modular_training_settings",
            type=str,
            help="Modular Training Settings",
            default=None,
        )
        self.parser.add_argument(
            "-nt", "--no_train", help="No Train", action="store_true", default=False
        )
        self.parser.add_argument(
            "-it", "--is_testing", help="Is Testing", action="store_true", default=False
        )
        self.parser.add_argument(
            "-of", "--overfitting", help="Overfitting", action="store_true", default=False
        )
        self.parser.add_argument(
            "-v", "--verbose", help="Verbose", action="store_true", default=False
        )
        self.parser.add_argument("-le", "--log_every", type=int, help="Log every", default=100)
        self.parser.add_argument(
            "-nc", "--no_continuity", help="No Continuity", action="store_true", default=False
        )
        self.parser.add_argument("-s", "--seed", type=int, help="Random Seed", default=0)
        self.parser.add_argument(
            "-smi", "--save_model_interval", type=int, help="Save Model Interval", default=1
        )
        self.parser.add_argument("-b", "--branch", type=str, help="Branch", default="")
        self.parser.add_argument("-f", "--focus", type=str, help="Focus", default=None)


class TestSessionArgParser(DefaultParser):
    """Set description, options, and flags for TestSession."""

    def __init__(self) -> None:
        """Define instance variables, collect arguments."""
        super().__init__("Test a protein sequence autoencoder")

    def _initialize(self) -> None:
        self.parser.add_argument("-n", "--model", type=str, help="Model Name", required=True)
        self.parser.add_argument(
            "-mv", "--model_version", type=str, help="Model Version", required=True
        )
        self.parser.add_argument("-mid", "--model_id", type=int, help="Model ID", required=True)
        # dataset0: seq:act_p, dataset1: seq:ss, dataset2: seq:act_p:ss
        self.parser.add_argument("-dcl", "--dataset_cl", type=str, help="Dataset_cl", default="")
        self.parser.add_argument("-dss", "--dataset_ss", type=str, help="Dataset_ss", default="")
        self.parser.add_argument(
            "-dclss", "--dataset_clss", type=str, help="Dataset_clss", default=""
        )
        # architecture blueprint
        self.parser.add_argument("-a", "--arch", type=str, help="Arch", required=True)
        # madel hyper params, d1: amino acids vec, dn: embedding, w: sliding window
        self.parser.add_argument("-d1", "--d1", type=int, help="D1", default=8)
        self.parser.add_argument("-dn", "--dn", type=int, help="Dn", default=10)
        self.parser.add_argument("-w", "--w", type=int, help="W", default=20)
        self.parser.add_argument("-teb", "--test_batch", type=int, help="Test Batch", default=1)
        self.parser.add_argument(
            "-v", "--verbose", help="Verbose", action="store_true", default=False
        )
        self.parser.add_argument(
            "-ge", "--get_embedding", help="Get Embedding", action="store_true", default=False
        )
        self.parser.add_argument("-tsne", "--tsne_dim", type=int, help="tSNE dim", default=0)
        self.parser.add_argument(
            "-eo", "--embed_only", help="Embed Only", action="store_true", default=False
        )
        self.parser.add_argument(
            "-se", "--smooth_embed", help="Smooth Embed", action="store_true", default=False
        )
        self.parser.add_argument("-prids", "--protein_ids", type=str, help="Pr IDs")
        self.parser.add_argument("-t", "--text", type=str, help="Text")
