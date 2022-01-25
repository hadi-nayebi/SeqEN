#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


from datetime import datetime
from os import system
from os.path import dirname
from pathlib import Path

from numpy import arange, mean
from pandas import DataFrame
from torch import cuda, device
from torch import save as torch_save

import wandb
from SeqEN2.autoencoder.adversarial_autoencoder import AdversarialAutoencoder
from SeqEN2.autoencoder.adversarial_autoencoder_classifier import (
    AdversarialAutoencoderClassifier,
)
from SeqEN2.autoencoder.adversarial_autoencoder_classifier_ss_decoder import (
    AdversarialAutoencoderClassifierSSDecoder,
)
from SeqEN2.autoencoder.autoencoder import Autoencoder
from SeqEN2.model.data_loader import DataLoader, write_json
from SeqEN2.utils.seq_tools import sliding_window


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
        self.data_loader_cl = None
        self.data_loader_ss = None
        self.data_loader_clss = None
        self.dataset_name_cl = None
        self.dataset_name_ss = None
        self.dataset_name_clss = None
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
        elif arch.type == "AAECSS":
            self.autoencoder = AdversarialAutoencoderClassifierSSDecoder(
                self.d1, self.dn, self.w, arch
            )
        self.autoencoder.to(self.device)

    def load_data(self, key, dataset_name):
        """
        Loading data once for a model to make sure the training/test sets are fixed.
        :param key:
        :param dataset_name:
        :return:
        """
        # load datafiles
        assert key in ["cl", "ss", "clss"], "unknown key for dataset"
        if key == "cl":
            self.data_loader_cl = DataLoader()
            self.data_loader_cl.load_test_data(dataset_name, self.device)
            self.data_loader_cl.load_train_data(dataset_name, self.device)
            self.dataset_name_cl = dataset_name
        elif key == "ss":
            self.data_loader_ss = DataLoader()
            self.data_loader_ss.load_test_data(dataset_name, self.device)
            self.data_loader_ss.load_train_data(dataset_name, self.device)
            self.dataset_name_ss = dataset_name
        elif key == "clss":
            self.data_loader_clss = DataLoader()
            self.data_loader_clss.load_test_data(dataset_name, self.device)
            self.data_loader_clss.load_train_data(dataset_name, self.device)
            self.dataset_name_clss = dataset_name

    def train(
        self,
        epochs=10,
        batch_size=128,
        test_interval=100,
        training_settings=None,
        input_noise=0.0,
        log_every=100,
        is_testing=False,
    ):
        assert self.data_loader_cl is not None, "at least dataset0 must be provided"
        if self.data_loader_ss is None and self.data_loader_clss is None:
            assert self.autoencoder.arch.type in ["AE", "AAE", "AAEC"], "wrong model type."
            self.train_AAEC(
                epochs=epochs,
                batch_size=batch_size,
                test_interval=test_interval,
                training_settings=training_settings,
                input_noise=input_noise,
                log_every=log_every,
                is_testing=is_testing,
            )
        else:
            assert self.autoencoder.arch.type == "AAECSS"
            if self.data_loader_clss is None:
                self.train_AAECSS_cl_ss(
                    epochs=epochs,
                    batch_size=batch_size,
                    test_interval=test_interval,
                    training_settings=training_settings,
                    input_noise=input_noise,
                    log_every=log_every,
                    is_testing=is_testing,
                )
            else:
                self.train_AAECSS_clss(
                    epochs=epochs,
                    batch_size=batch_size,
                    test_interval=test_interval,
                    training_settings=training_settings,
                    input_noise=input_noise,
                    log_every=log_every,
                    is_testing=is_testing,
                )

    def get_train_batch_cl(self, batch_size):
        for batch in self.data_loader_cl.get_train_batch(batch_size=batch_size):
            yield batch

    def get_train_batch_cl_ss(self, batch_size, max_size=None):
        for batch_cl, batch_ss in zip(
            self.data_loader_cl.get_train_batch(batch_size=batch_size, max_size=max_size),
            self.data_loader_ss.get_train_batch(batch_size=batch_size, max_size=max_size),
        ):
            yield {"cl": batch_cl, "ss": batch_ss}

    def get_train_batch_clss(self, batch_size, max_size=None):
        for batch_cl, batch_ss, batch_clss in zip(
            self.data_loader_cl.get_train_batch(batch_size=batch_size, max_size=max_size),
            self.data_loader_ss.get_train_batch(batch_size=batch_size, max_size=max_size),
            self.data_loader_clss.get_train_batch(batch_size=batch_size, max_size=max_size),
        ):
            yield {"cl": batch_cl, "ss": batch_ss, "clss": batch_clss}

    def log_it(self, iter, epoch):
        wandb.log({"epoch": epoch, "iter": iter})
        for key, item in self.autoencoder.logs.items():
            wandb.log({key: wandb.Histogram(item)})
            wandb.log({f"{key}_mean": mean(item)})
        self.autoencoder.reset_log()

    def initialize_training(
        self,
        batch_size=128,
        training_settings=None,
        input_noise=0.0,
    ):
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
        self.config.dataset_name_cl = self.dataset_name_cl
        self.autoencoder.initialize_for_training(training_settings=training_settings)
        self.config.training_settings = self.autoencoder.training_settings.to_dict()
        self.config.model_type = model_type
        self.config.arch = arch_name
        wandb.watch(self.autoencoder)
        return train_dir

    def store_model(self, model, train_dir, epoch):
        model_path = str(train_dir / f"epoch_{epoch}.model")
        torch_save(self.autoencoder, model_path)
        model.add_file(model_path)
        self.autoencoder.save(train_dir, epoch)
        self.autoencoder.save_training_settings(train_dir)

    def finalize_training(self, train_dir, is_testing=False):
        if is_testing:
            system(f"rm -r {train_dir}")
        else:
            system(f"mv {str(train_dir)} {str(train_dir)}_done")

    def train_AAEC(
        self,
        epochs=10,
        batch_size=128,
        test_interval=100,
        training_settings=None,
        input_noise=0.0,
        log_every=100,
        is_testing=False,
    ):
        train_dir = self.initialize_training(
            batch_size=batch_size,
            training_settings=training_settings,
            input_noise=input_noise,
        )
        model = wandb.Artifact(f"{self.name}_model", type="model")
        # start training loop
        iter_for_test = 0
        iter_for_log = 0
        for epoch in range(0, epochs):
            for batch in self.get_train_batch_cl(batch_size):
                self.autoencoder.train_batch(batch, self.device, input_noise=input_noise)
                iter_for_test += 1
                iter_for_log += 1
                if iter_for_test == test_interval:
                    iter_for_test = 0
                    self.test_AAEC()
                if (iter_for_log + 1) % log_every == 0:
                    self.log_it(iter_for_log, epoch)
            self.store_model(model, train_dir, epoch)
            self.autoencoder.update_training_settings(train_dir)
        self.finalize_training(train_dir, is_testing=is_testing)

    def test_AAEC(self, num_test_items=1):
        for test_batch, metadata in self.data_loader_cl.get_test_batch(batch_size=num_test_items):
            # using metadata?
            self.autoencoder.test_batch(test_batch, self.device)

    def train_AAECSS_cl_ss(
        self,
        epochs=10,
        batch_size=128,
        test_interval=100,
        training_settings=None,
        input_noise=0.0,
        log_every=100,
        is_testing=False,
    ):
        train_dir = self.initialize_training(
            batch_size=batch_size,
            training_settings=training_settings,
            input_noise=input_noise,
        )
        model = wandb.Artifact(f"{self.name}_model", type="model")
        # start training loop
        iter_for_test = 0
        iter_for_log = 0
        # for training
        max_size = max(self.data_loader_cl.train_data_size, self.data_loader_ss.train_data_size)

        for epoch in range(0, epochs):

            for batch in self.get_train_batch_cl_ss(batch_size, max_size=max_size):
                self.autoencoder.train_batch(batch, self.device, input_noise=input_noise)
                iter_for_test += 1
                iter_for_log += 1
                if iter_for_test == test_interval:
                    iter_for_test = 0
                    self.test_AAECSS_cl_ss()
                if (iter_for_log + 1) % log_every == 0:
                    self.log_it(iter_for_log, epoch)
            self.store_model(model, train_dir, epoch)
            self.autoencoder.update_training_settings(train_dir)
        self.finalize_training(train_dir, is_testing=is_testing)

    def test_AAECSS_cl_ss(self, num_test_items=1):
        for (batch_cl, metadata_cl), (batch_ss, metadata_ss) in zip(
            self.data_loader_cl.get_test_batch(batch_size=num_test_items),
            self.data_loader_ss.get_test_batch(batch_size=num_test_items),
        ):
            # using metadata?
            test_batch = {"cl": batch_cl, "ss": batch_ss}
            self.autoencoder.test_batch(test_batch, self.device)

    def train_AAECSS_clss(
        self,
        epochs=10,
        batch_size=128,
        test_interval=100,
        training_settings=None,
        input_noise=0.0,
        log_every=100,
        is_testing=False,
    ):
        train_dir = self.initialize_training(
            batch_size=batch_size,
            training_settings=training_settings,
            input_noise=input_noise,
        )
        model = wandb.Artifact(f"{self.name}_model", type="model")
        # start training loop
        iter_for_test = 0
        iter_for_log = 0
        max_size = max(
            self.data_loader_cl.train_data_size,
            self.data_loader_ss.train_data_size,
            self.data_loader_clss.train_data_size,
        )
        for epoch in range(0, epochs):
            for batch in self.get_train_batch_clss(batch_size, max_size=max_size):
                self.autoencoder.train_batch(batch, self.device, input_noise=input_noise)
                iter_for_test += 1
                iter_for_log += 1
                if iter_for_test == test_interval:
                    iter_for_test = 0
                    self.test_AAECSS_clss()
                if (iter_for_log + 1) % log_every == 0:
                    self.log_it(iter_for_log, epoch)
            self.store_model(model, train_dir, epoch)
            self.autoencoder.update_training_settings(train_dir)
        self.finalize_training(train_dir, is_testing=is_testing)

    def test_AAECSS_clss(self, num_test_items=1):
        for batch_cl, batch_ss, batch_clss in zip(
            self.data_loader_cl.get_test_batch(batch_size=num_test_items),
            self.data_loader_ss.get_test_batch(batch_size=num_test_items),
            self.data_loader_clss.get_test_batch(batch_size=num_test_items),
        ):
            test_batch = {"cl": batch_cl, "ss": batch_ss, "clss": batch_clss}
            self.autoencoder.test_batch(test_batch, self.device)

    def overfit(self, epochs=1000, num_test_items=1, input_noise=0.0, training_settings=None):
        raise NotImplementedError("No longer using this method, need a redo")
        # wandb.init(project=self.name, name="overfitting")
        # self.config = wandb.config
        # self.config.batch_size = num_test_items
        # self.config.input_noise = input_noise
        # self.config.dataset_name = self.dataset_name
        # self.autoencoder.initialize_for_training(training_settings)
        # self.config.training_settings = self.autoencoder.training_settings
        # wandb.watch(self.autoencoder)
        # for epoch in range(0, epochs):
        #     wandb.log({"epoch": epoch})
        #     for overfit_batch in self.data_loader.get_test_batch(
        #         num_test_items=num_test_items, random=False
        #     ):
        #         self.autoencoder.train_batch(overfit_batch, self.device, input_noise=input_noise)
        #         self.autoencoder.test_batch(overfit_batch, self.device)

    def load_model(self, version, model_id):
        model_dir = self.root / "models" / self.name / "versions" / version
        self.autoencoder.load(model_dir, model_id)

    def get_embedding(self, num_test_items=-1):
        for input_vals, metadata in self.data_loader_cl.get_test_batch(batch_size=num_test_items):
            embedding, classifier_output, consensus_ss = self.autoencoder.embed_batch(
                input_vals, self.device
            )
            new_df = DataFrame([])
            new_df.attrs["name"] = metadata["name"]
            new_df.attrs["seq_ndx"] = input_vals[:, 0]
            new_df.attrs["trg_act"] = input_vals[:, 1]
            new_df.attrs["cons_ss"] = consensus_ss
            new_df["unique_id"] = arange(classifier_output[:, 0].shape[0])
            new_df["act_pred"] = classifier_output[:, 0].cpu()
            new_df["act_trg"] = (
                sliding_window(input_vals[:, 1].reshape((-1, 1)), self.w).mean(axis=1).cpu()
            )
            new_df["slices"] = sliding_window(
                input_vals[:, 0].reshape((-1, 1)), self.w, keys="WYFMILVAGPSTCEDQNHRK*"
            )
            new_df["embedding"] = embedding.tolist()
            yield new_df
