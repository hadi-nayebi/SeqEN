#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


from os import system
from os.path import dirname
from pathlib import Path
from typing import Dict

from SeqEN2.autoencoder.utils import Architecture
from SeqEN2.model.data_loader import read_json
from SeqEN2.model.model import Model
from SeqEN2.utils.custom_arg_parser import TrainSessionArgParser
from SeqEN2.utils.utils import set_random_seed


class TrainSession:

    root = Path(dirname(__file__)).parent.parent

    def __init__(self, is_testing=False):
        self.is_testing = is_testing
        # setup dirs
        self.models_dir = self.root / "models"
        if not self.models_dir.exists():
            self.models_dir.mkdir()
        self.data_dir = self.root / "data"
        if not self.data_dir.exists():
            self.data_dir.mkdir()
        self.arch_dir = self.root / "config" / "arch"
        if not self.arch_dir.exists():
            self.arch_dir.mkdir()
        self.train_params_dir = self.root / "config" / "train_params"
        if not self.train_params_dir.exists():
            self.train_params_dir.mkdir()

        # model placeholder
        self.model = None

    def add_model(self, name, arch, d1=8, dn=10, w=20):
        arch = self.load_arch(arch)
        if self.model is None:
            self.model = Model(name, arch, d1=d1, dn=dn, w=w)

    def load_data(self, key, dataset_name):
        self.model.load_data(key, dataset_name)

    def load_arch(self, arch):
        arch_path = self.arch_dir / f"{arch}.json"
        return Architecture(read_json(str(arch_path)))

    def load_training_settings(self, training_settings=None) -> Dict:
        if training_settings is not None:
            if isinstance(training_settings, Path):
                training_settings = read_json(str(training_settings))
            elif isinstance(training_settings, str):
                training_settings_path = self.train_params_dir / f"{training_settings}.json"
                training_settings = read_json(str(training_settings_path))
            else:
                raise TypeError("Train setting must be str or Path-like object")
        return training_settings

    def train(
        self,
        epochs=10,
        batch_size=128,
        test_interval=100,
        training_settings=None,
        input_noise=0.0,
        log_every=100,
        mvid=None,
        ignore_continuity=False,
    ):
        training_settings = self.load_training_settings(training_settings)
        if self.is_testing:
            # add more default setting for is_testing
            epochs = 1
        self.model.train(
            epochs=epochs,
            batch_size=batch_size,
            test_interval=test_interval,
            training_settings=training_settings,
            input_noise=input_noise,
            log_every=log_every,
            is_testing=self.is_testing,
            mvid=mvid,
            ignore_continuity=ignore_continuity,
        )

    def test(self, num_test_items=1):
        self.model.test(num_test_items=num_test_items)

    def overfit_tests(self, epochs=1000, num_test_items=1, input_noise=0.0, training_settings=None):
        # overfit single sequence
        self.model.overfit(
            epochs=epochs,
            num_test_items=num_test_items,
            input_noise=input_noise,
            training_settings=training_settings,
        )


def main(args):
    # session
    set_random_seed(args["Random Seed"])
    train_session = TrainSession(is_testing=args["Is Testing"])
    train_session.add_model(
        args["Model Name"],
        args["Arch"],
        d1=args["D1"],
        dn=args["Dn"],
        w=args["W"],
    )
    # load datafiles
    mvid = None
    training_settings = args["Training Settings"]
    if args["Dataset_cl"] != "":
        train_session.load_data("cl", args["Dataset_cl"])
    if args["Dataset_ss"] != "":
        train_session.load_data("ss", args["Dataset_ss"])
    if args["Dataset_clss"] != "":
        train_session.load_data("clss", args["Dataset_clss"])
    if args["Model Version ID"] != "" and args["Model Version ID"] != "x":
        version, model_id = args["Model Version ID"].split("#")
        train_session.model.load_model(version, model_id)
        mvid = [version.split("_")[0]] + [model_id]
        training_settings = train_session.model.versions_path / version / "training_settings.json"
    if args["Overfitting"]:
        train_session.overfit_tests(
            epochs=args["Epochs"],
            num_test_items=args["Test Batch"],
            input_noise=args["Input Noise"],
            training_settings=args["Training Settings"],
        )
    elif args["No Train"]:
        train_session.test(num_test_items=args["Test Batch"])
    else:
        train_session.train(
            epochs=args["Epochs"],
            batch_size=args["Train Batch"],
            test_interval=args["Test Interval"],
            training_settings=training_settings,
            input_noise=args["Input Noise"],
            log_every=args["Log every"],
            mvid=mvid,
            ignore_continuity=args["No Continuity"],
        )


if __name__ == "__main__":
    # parse arguments
    parser = TrainSessionArgParser()
    parsed_args = parser.parsed()
    system("wandb login")
    main(parsed_args)

# examples :
# ./SeqEN2/sessions/train_session.py -n dummy -dcl kegg_ndx_ACTp -dss pdb_ndx_ss -dclss pdb_act_clss -a arch8 -e 2
# -trb 10 -no 0.05 -le 10
