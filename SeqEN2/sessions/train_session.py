#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


from os import system
from os.path import dirname
from pathlib import Path

from SeqEN2.autoencoder.utils import Architecture
from SeqEN2.model.data_loader import read_json
from SeqEN2.model.model import Model
from SeqEN2.utils.custom_arg_parser import TrainSessionArgParser


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

    def add_model(self, name, arch, model_type, d1=8, dn=10, w=20):
        arch = self.load_arch(arch)
        if self.model is None:
            self.model = Model(name, arch, model_type, d1=d1, dn=dn, w=w)

    def load_data(self, dataset_name):
        self.model.load_data(dataset_name)

    def load_arch(self, arch):
        arch_path = self.root / "config" / "arch" / f"{arch}.json"
        return Architecture(read_json(str(arch_path)))

    def load_train_params(self, train_params=None):
        if train_params is not None:
            train_params_path = self.root / "config" / "train_params" / f"{train_params}.json"
            train_params = read_json(str(train_params_path))
        return train_params

    def train(
        self,
        run_title,
        epochs=10,
        batch_size=128,
        test_interval=100,
        training_params=None,
        input_noise=0.0,
        log_every=100,
    ):
        if self.is_testing:
            epochs = 1
        training_params = self.load_train_params(training_params)
        self.model.train(
            run_title,
            epochs=epochs,
            batch_size=batch_size,
            test_interval=test_interval,
            training_params=training_params,
            input_noise=input_noise,
            log_every=100,
        )

    def test(self, num_test_items=1):
        self.model.test(num_test_items=num_test_items)

    def overfit_tests(self, epochs=1000, num_test_items=1, input_noise=0.0, training_params=None):
        # overfit single sequence
        self.model.overfit(
            f"overfit_{num_test_items}",
            epochs=epochs,
            num_test_items=num_test_items,
            input_noise=input_noise,
            training_params=training_params,
        )


def main(args):
    # session
    train_session = TrainSession(is_testing=args["Is Testing"])
    train_session.add_model(
        args["Model Name"],
        args["Arch"],
        args["Model Type"],
        d1=args["D1"],
        dn=args["Dn"],
        w=args["W"],
    )
    # load datafiles
    train_session.load_data(args["Dataset"])
    # if args['Model ID'] != '':
    #     session.model.load_model(args['Model ID'], map_location=get_map_location())
    if args["Overfitting"]:
        train_session.overfit_tests(
            epochs=args["Epochs"],
            num_test_items=args["Test Batch"],
            input_noise=args["Input Noise"],
            training_params=args["Train Params"],
        )
    elif args["No Train"]:
        train_session.test(num_test_items=args["Test Batch"])
    else:
        train_session.train(
            args["Run Title"],
            epochs=args["Epochs"],
            batch_size=args["Train Batch"],
            test_interval=args["Test Interval"],
            training_params=args["Train Params"],
            input_noise=args["Input Noise"],
            log_every=args["Log every"],
        )
    if train_session.is_testing:
        train_dir = train_session.model.versions_path / args["Run Title"]
        system(f"rm -r {train_dir}")


if __name__ == "__main__":
    # parse arguments
    parser = TrainSessionArgParser()
    parsed_args = parser.parsed()
    system("wandb login")
    main(parsed_args)
