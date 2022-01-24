#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from typing import Dict

from numpy.random import choice
from torch import Tensor, argmax
from torch import load as torch_load
from torch import no_grad, optim, randperm
from torch import save as torch_save
from torch import sum as torch_sum
from torch import tensor, transpose
from torch.nn import Module, NLLLoss
from torch.nn.functional import one_hot, unfold

import wandb
from SeqEN2.autoencoder.utils import CustomLRScheduler, LayerMaker
from SeqEN2.model.data_loader import write_json
from SeqEN2.utils.custom_dataclasses import AETrainingSettings
from SeqEN2.utils.seq_tools import consensus_acc
from SeqEN2.utils.utils import get_map_location


# class for AE
class Autoencoder(Module):

    aa_keys = "WYFMILVAGPSTCEDQNHRK*"  # amino acids class labels
    d0 = 21  # amino acids class size

    def __init__(self, d1, dn, w, arch):
        super(Autoencoder, self).__init__()
        # common attr
        self.d1 = d1
        self.dn = dn
        self.w = w
        self.arch = arch
        # Modules
        self.vectorizer = LayerMaker().make(self.arch.vectorizer)
        self.encoder = LayerMaker().make(self.arch.encoder)
        self.decoder = LayerMaker().make(self.arch.decoder)
        self.devectorizer = LayerMaker().make(self.arch.devectorizer)
        # training components
        self._training_settings = AETrainingSettings()
        # define customized optimizers
        self.reconstructor_optimizer = None
        self.reconstructor_lr_scheduler = None
        # Loss functions
        self.criterion_NLLLoss = NLLLoss()
        # logger
        self.logs = {}

    @property
    def training_settings(self) -> AETrainingSettings:
        return self._training_settings

    @training_settings.setter
    def training_settings(self, value=None) -> None:
        if isinstance(value, Dict) or value is None or isinstance(value, AETrainingSettings):
            if isinstance(value, Dict):
                try:
                    self._training_settings = AETrainingSettings(**value)
                except TypeError as e:
                    raise KeyError(f"missing/extra keys for AETrainingSettings, {e}")
            elif isinstance(value, AETrainingSettings):
                self._training_settings = value
        else:
            raise TypeError(
                f"Training settings must be a dict or None or type AETrainingSettings, {type(value)} is passed."
            )

    def save_training_settings(self, train_dir):
        write_json(
            self.training_settings.to_dict(),
            str(train_dir / "training_settings.json"),
        )

    def forward_encoder_decoder(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        decoded = transpose(self.decoder(encoded), 1, 2).reshape(-1, self.d1)
        devectorized = self.devectorizer(decoded)
        return devectorized

    def forward_test(self, one_hot_input):
        return self.forward_encoder_decoder(one_hot_input)

    def save(self, model_dir, epoch):
        torch_save(self.vectorizer, model_dir / f"vectorizer_{epoch}.m")
        torch_save(self.encoder, model_dir / f"encoder_{epoch}.m")
        torch_save(self.decoder, model_dir / f"decoder_{epoch}.m")
        torch_save(self.devectorizer, model_dir / f"devectorizer_{epoch}.m")

    def load(self, model_dir, model_id):
        self.vectorizer = torch_load(
            model_dir / f"vectorizer_{model_id}.m", map_location=get_map_location()
        )
        self.encoder = torch_load(
            model_dir / f"encoder_{model_id}.m", map_location=get_map_location()
        )
        self.decoder = torch_load(
            model_dir / f"decoder_{model_id}.m", map_location=get_map_location()
        )
        self.devectorizer = torch_load(
            model_dir / f"devectorizer_{model_id}.m", map_location=get_map_location()
        )

    def initialize_training_components(self):
        # define customized optimizers
        self.reconstructor_optimizer = optim.SGD(
            [
                {"params": self.vectorizer.parameters()},
                {"params": self.encoder.parameters()},
                {"params": self.decoder.parameters()},
                {"params": self.devectorizer.parameters()},
            ],
            lr=self._training_settings.reconstructor.lr,
        )
        self.reconstructor_lr_scheduler = CustomLRScheduler(
            self.reconstructor_optimizer,
            factor=self._training_settings.reconstructor.factor,
            patience=self._training_settings.reconstructor.patience,
            min_lr=self._training_settings.reconstructor.min_lr,
        )

    def initialize_for_training(self, training_settings=None):
        self.training_settings = training_settings
        self.initialize_training_components()

    def transform_input(self, input_vals, device, input_noise=0.0):
        # scans by sliding window of w
        assert isinstance(input_vals, Tensor)
        kernel_size = (input_vals.shape[1], self.w)
        input_vals = unfold(input_vals.T[None, None, :, :], kernel_size=kernel_size)[0].T
        input_ndx = input_vals[:, : self.w].long()
        one_hot_input = one_hot(input_ndx, num_classes=self.d0) * 1.0
        if input_noise > 0.0:
            ndx = randperm(self.w)
            size = list(one_hot_input.shape)
            size[-1] = 1
            p = tensor(choice([1, 0], p=[input_noise, 1 - input_noise], size=size)).to(device)
            mutated_one_hot = (one_hot_input[:, ndx, :] * p) + (one_hot_input * (1 - p))
            return input_ndx, mutated_one_hot
        else:
            return input_ndx, one_hot_input

    def log(self, key, value):
        if key in self.logs.keys():
            self.logs[key].append(value)
        else:
            self.logs[key] = [value]

    def reset_log(self):
        self.logs = {}

    def train_batch(self, input_vals, device, input_noise=0.0):
        """
        Training for one batch of data, this will move into autoencoder module
        :param input_vals:
        :param device:
        :param input_noise:
        :return:
        """
        self.train()
        input_ndx, one_hot_input = self.transform_input(input_vals, device, input_noise=input_noise)
        # train encoder_decoder
        self.reconstructor_optimizer.zero_grad()
        reconstructor_output = self.forward_encoder_decoder(one_hot_input)
        reconstructor_loss = self.criterion_NLLLoss(reconstructor_output, input_ndx.reshape((-1,)))
        reconstructor_loss.backward()
        self.reconstructor_optimizer.step()
        self.log("reconstructor_loss", reconstructor_loss.item())
        self.log("reconstructor_LR", self.reconstructor_lr_scheduler.get_last_lr())
        self._training_settings.reconstructor.lr = self.reconstructor_lr_scheduler.get_last_lr()
        self.reconstructor_lr_scheduler.step(reconstructor_loss.item())
        # clean up
        del input_ndx
        del one_hot_input
        del reconstructor_loss
        del reconstructor_output

    def test_batch(self, input_vals, device):
        """
        Test a single batch of data, this will move into autoencoder
        :param input_vals:
        :return:
        """
        self.eval()
        with no_grad():
            input_ndx, one_hot_input = self.transform_input(input_vals, device, input_noise=0.0)
            reconstructor_output = self.forward_test(one_hot_input)
            reconstructor_loss = self.criterion_NLLLoss(
                reconstructor_output, input_ndx.reshape((-1,))
            )
            # reconstructor acc
            reconstructor_ndx = argmax(reconstructor_output, dim=1)
            reconstructor_accuracy = (
                torch_sum(reconstructor_ndx == input_ndx.reshape((-1,)))
                / reconstructor_ndx.shape[0]
            )
            consensus_seq_acc, _ = consensus_acc(
                input_ndx, reconstructor_ndx.reshape((-1, self.w)), device
            )
            # reconstruction_loss, discriminator_loss, classifier_loss
            self.log("test_reconstructor_loss", reconstructor_loss.item())
            self.log("test_reconstructor_accuracy", reconstructor_accuracy.item())
            self.log("test_consensus_accuracy", consensus_seq_acc)
            # clean up
            del input_ndx
            del one_hot_input
            del reconstructor_loss
            del reconstructor_output
