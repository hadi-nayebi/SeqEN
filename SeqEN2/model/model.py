#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


from glob import glob
from os.path import dirname
from pathlib import Path

from torch import cuda, device
from torch import save as torch_save

import wandb
from SeqEN2.autoencoder.adversarial_autoencoder import (
    AdversarialAutoencoder,
    AdversarialAutoencoderClassifier,
    Autoencoder,
)
from SeqEN2.utils.data_loader import DataLoader, write_json


class Model:
    """
    The Model object contains the ML unit and training dataset
    """

    root = Path(dirname(__file__)).parent.parent

    def __init__(self, name, arch, model_type, d0=21, d1=8, dn=10, w=20):
        self.name = name
        self.path = self.root / "models" / f"{self.name}"
        self.versions_path = self.path / "versions"
        self.d0 = d0
        self.d1 = d1
        self.dn = dn
        self.w = w
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.autoencoder = None
        self.build_model(model_type, arch)
        self.data = {"train_data": None, "test_data": None}
        self.data_loader = None
        self.dataset_name = None
        self.config = None
        if not self.path.exists():
            self.path.mkdir()
            self.versions_path.mkdir()

    def build_model(self, model_type, arch):
        if model_type == "AE":
            self.autoencoder = Autoencoder(self.d0, self.d1, self.dn, self.w, arch)
        elif model_type == "AAE":
            self.autoencoder = AdversarialAutoencoder(
                self.d0, self.d1, self.dn, self.w, arch
            )
        elif model_type == "AAEC":
            self.autoencoder = AdversarialAutoencoderClassifier(
                self.d0, self.d1, self.dn, self.w, arch
            )
        self.autoencoder.to(self.device)

    def load_data(self, dataset_name, datasets):
        """
        Loading data once for a model to make sure the training/test sets are fixed.
        :param dataset_name:
        :param datasets:
        :return:
        """
        if self.dataset_name is None:
            self.dataset_name = dataset_name
        # load datafiles
        num_files = len(datasets)
        test_data_files = max(1, num_files // 10)
        train_data = datasets[test_data_files:]
        test_data = datasets[:test_data_files]
        if self.data["test_data"] is None:
            self.data["test_data"] = test_data
        if self.data["train_data"] is None:
            self.data["train_data"] = train_data
        self.data_loader = DataLoader(self.data["train_data"], self.data["test_data"])

    def train(
        self,
        run_title,
        epochs=10,
        batch_size=128,
        num_test_items=1,
        test_interval=100,
        training_params=None,
        input_noise=0.0,
    ):
        """
        The main training loop for a model
        :param run_title:
        :param epochs:
        :param batch_size:
        :param num_test_items:
        :param test_interval:
        :param training_params:
        :param input_noise:
        :return:
        """
        wandb.init(project=self.name, name=run_title)
        self.config = wandb.config
        self.config.batch_size = batch_size
        self.config.input_noise = input_noise
        self.config.dataset_name = self.dataset_name
        self.autoencoder.initialize_for_training(training_params)
        self.config.training_params = self.autoencoder.training_params
        wandb.watch(self.autoencoder)
        model = wandb.Artifact(f"{self.name}_model", type="model")
        train_dir = self.versions_path / f"{run_title}"
        if not train_dir.exists():
            train_dir.mkdir()
        else:
            print("Choose a different title for the run!")
            return
        iter_for_test = 0
        for epoch in range(0, epochs):
            wandb.log({"epoch": epoch})
            for batch in self.data_loader.get_train_batch(batch_size=batch_size):
                self.autoencoder.train_batch(
                    batch, self.device, input_noise=input_noise
                )
                iter_for_test += 1
                if iter_for_test == test_interval:
                    iter_for_test = 0
                    for test_batch in self.data_loader.get_test_batch(
                        num_test_items=num_test_items
                    ):
                        self.autoencoder.test_batch(
                            test_batch, self.device, input_noise=input_noise
                        )
            model_path = str(train_dir / f"epoch_{epoch}.model")
            torch_save(self.autoencoder, model_path)
            model.add_file(model_path)
            self.autoencoder.save(train_dir, epoch)

        write_json(
            self.autoencoder.training_params,
            str(train_dir / f"{run_title}_train_params.json"),
        )
        write_json(self.data, str(train_dir / f"{run_title}_data.json"))

    # def load_model(self, model_id, map_location):
    #     version, model_name, run_title = model_id.split(',')          # 0,test,run_title
    #     try:
    #         model_dir = self.root / 'models' / model_name / 'versions' / run_title
    #         self.autoencoder.load(model_dir, version, map_location=map_location)
    #         print('first method is working')
    #     except FileNotFoundError:
    #         model_dir = Path('/mnt/home/nayebiga/SeqEncoder/SeqEN/models') / model_name / 'versions' / run_title
    #         self.autoencoder.load(model_dir, version, map_location=map_location)
    #         print('second method is working')
