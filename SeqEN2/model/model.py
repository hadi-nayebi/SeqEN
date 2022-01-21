#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


from datetime import datetime
from os import system
from os.path import dirname
from pathlib import Path

from torch import cuda, device
from torch import save as torch_save

import wandb
from SeqEN2.autoencoder.adversarial_autoencoder import AdversarialAutoencoder
from SeqEN2.autoencoder.adversarial_autoencoder_classifier import (
    AdversarialAutoencoderClassifier,
)
from SeqEN2.autoencoder.autoencoder import Autoencoder
from SeqEN2.model.data_loader import DataLoader, write_json


class Model:
    """
    The Model object contains the ML unit and training dataset
    """

    root = Path(dirname(__file__)).parent.parent

    def __init__(self, name, arch, d1=8, dn=10, w=20):
        self.name = name
        self.path = self.root / "models" / f"{self.name}"
        self.versions_path = self.path / "versions"
        self.d1 = d1
        self.dn = dn
        self.w = w
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.autoencoder = None
        self.build_model(arch)
        self.data_loader = None
        self.dataset_name = None
        self.config = None
        if not self.path.exists():
            self.path.mkdir()
            self.versions_path.mkdir()

    def build_model(self, arch):
        if arch.type == "AE":
            self.autoencoder = Autoencoder(self.d1, self.dn, self.w, arch)
        elif arch.type == "AAE":
            self.autoencoder = AdversarialAutoencoder(self.d1, self.dn, self.w, arch)
        elif arch.type == "AAEC":
            self.autoencoder = AdversarialAutoencoderClassifier(self.d1, self.dn, self.w, arch)
        self.autoencoder.to(self.device)

    def load_data(self, dataset_name):
        """
        Loading data once for a model to make sure the training/test sets are fixed.
        :param dataset_name:
        :return:
        """
        # load datafiles
        self.data_loader = DataLoader()
        self.data_loader.load_test_data(dataset_name, self.device)
        self.data_loader.load_train_data(dataset_name, self.device)

    def train(
        self,
        epochs=10,
        batch_size=128,
        test_interval=100,
        training_params=None,
        input_noise=0.0,
        log_every=100,
        is_testing=False,
    ):
        """
        The main training loop for a model
        :param epochs:
        :param batch_size:
        :param test_interval:
        :param training_params:
        :param input_noise:
        :param log_every:
        :return:
        """
        now = datetime.now().strftime("%Y%m%d%H%M")
        model_type = self.autoencoder.arch.type
        arch_name = self.autoencoder.arch.name
        run_title = f"{now}_{model_type}_{arch_name}"
        train_dir = self.versions_path / f"{run_title}"
        assert (
            not train_dir.exists()
        ), "This directory already exist, choose a different title for the run!"
        train_dir.mkdir()
        # connect to wandb
        wandb.init(project=self.name, name=run_title)
        self.config = wandb.config
        self.config.batch_size = batch_size
        self.config.input_noise = input_noise
        self.config.dataset_name = self.dataset_name
        self.autoencoder.initialize_for_training(training_params)
        self.config.training_params = self.autoencoder.training_params
        self.config.model_type = model_type
        self.config.arch = arch_name
        wandb.watch(self.autoencoder)
        model = wandb.Artifact(f"{self.name}_model", type="model")
        # start training loop
        iter_for_test = 0
        iter_for_log = 0
        for epoch in range(0, epochs):
            for batch in self.data_loader.get_train_batch(batch_size=batch_size):
                self.autoencoder.train_batch(batch, self.device, input_noise=input_noise)
                iter_for_test += 1
                iter_for_log += 1
                if iter_for_test == test_interval:
                    iter_for_test = 0
                    self.test()
                if (iter_for_log + 1) % log_every == 0:
                    wandb.log({"epoch": epoch, "iter": iter_for_log})
                    for key, item in self.autoencoder.logs.items():
                        wandb.log({key: wandb.Histogram(item), "iter": iter_for_log})
                    self.autoencoder.reset_log()
            model_path = str(train_dir / f"epoch_{epoch}.model")
            torch_save(self.autoencoder, model_path)
            model.add_file(model_path)
            self.autoencoder.save(train_dir, epoch)
            write_json(
                self.autoencoder.training_params,
                str(train_dir / f"{run_title}_train_params.json"),
            )
        if is_testing:
            system(f"rm -r {train_dir}")
        else:
            system(f"mv {str(train_dir)} {str(train_dir)}_done")

    def test(self, num_test_items=1):
        """
        The main training loop for a model
        :param num_test_items:
        :return:
        """
        for test_batch, metadata in self.data_loader.get_test_batch(batch_size=num_test_items):
            # using metadata?
            self.autoencoder.test_batch(test_batch, self.device)

    def overfit(self, epochs=1000, num_test_items=1, input_noise=0.0, training_params=None):
        raise NotImplementedError("No longer using this method, need a redo")
        # wandb.init(project=self.name, name="overfitting")
        # self.config = wandb.config
        # self.config.batch_size = num_test_items
        # self.config.input_noise = input_noise
        # self.config.dataset_name = self.dataset_name
        # self.autoencoder.initialize_for_training(training_params)
        # self.config.training_params = self.autoencoder.training_params
        # wandb.watch(self.autoencoder)
        # for epoch in range(0, epochs):
        #     wandb.log({"epoch": epoch})
        #     for overfit_batch in self.data_loader.get_test_batch(
        #         num_test_items=num_test_items, random=False
        #     ):
        #         self.autoencoder.train_batch(overfit_batch, self.device, input_noise=input_noise)
        #         self.autoencoder.test_batch(overfit_batch, self.device)

    def load_model(self, model_id, map_location):
        raise NotImplementedError("needs a redo")
        # version, model_name, run_title = model_id.split(',')          # 0,test,run_title
        # try:
        #     model_dir = self.root / 'models' / model_name / 'versions' / run_title
        #     self.autoencoder.load(model_dir, version, map_location=map_location)
        #     print('first method is working')
        # except FileNotFoundError:
        #     model_dir = Path('/mnt/home/nayebiga/SeqEncoder/SeqEN/models') / model_name / 'versions' / run_title
        #     self.autoencoder.load(model_dir, version, map_location=map_location)
        #     print('second method is working')
