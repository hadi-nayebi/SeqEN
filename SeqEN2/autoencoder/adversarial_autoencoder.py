#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from typing import Dict

from torch import argmax
from torch import load as torch_load
from torch import no_grad, ones, optim, randperm
from torch import save as torch_save
from torch import sum as torch_sum
from torch import transpose, zeros

import wandb
from SeqEN2.autoencoder.autoencoder import Autoencoder
from SeqEN2.autoencoder.utils import CustomLRScheduler, LayerMaker
from SeqEN2.utils.custom_dataclasses import AAETrainingSettings
from SeqEN2.utils.seq_tools import consensus_acc
from SeqEN2.utils.utils import get_map_location


# class for AAE
class AdversarialAutoencoder(Autoencoder):
    def __init__(self, d1, dn, w, arch):
        super(AdversarialAutoencoder, self).__init__(d1, dn, w, arch)
        assert self.arch.discriminator is not None, "arch missing discriminator."
        self.discriminator = LayerMaker().make(self.arch.discriminator)
        # training components
        self._training_settings = AAETrainingSettings()
        # define customized optimizers
        self.generator_optimizer = None
        self.generator_lr_scheduler = None
        ###
        self.discriminator_optimizer = None
        self.discriminator_lr_scheduler = None

    @property
    def training_settings(self) -> AAETrainingSettings:
        return self._training_settings

    @training_settings.setter
    def training_settings(self, value=None) -> None:
        if isinstance(value, Dict) or value is None or isinstance(value, AAETrainingSettings):
            if isinstance(value, Dict):
                try:
                    self._training_settings = AAETrainingSettings(**value)
                except TypeError as e:
                    raise KeyError(f"missing/extra keys for AAETrainingSettings, {e}")
            elif isinstance(value, AAETrainingSettings):
                self._training_settings = value
        else:
            raise TypeError(
                f"Training settings must be a dict or None or type AAETrainingSettings, {type(value)} is passed."
            )

    def forward_generator(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        discriminator_output = self.discriminator(encoded)
        return discriminator_output

    def forward_discriminator(self, one_hot_input):
        return self.forward_generator(one_hot_input)

    def forward_test(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        decoded = transpose(self.decoder(encoded), 1, 2).reshape(-1, self.d1)
        devectorized = self.devectorizer(decoded)
        discriminator_output = self.discriminator(encoded)
        return devectorized, discriminator_output

    def save(self, model_dir, epoch):
        super(AdversarialAutoencoder, self).save(model_dir, epoch)
        torch_save(self.discriminator, model_dir / f"discriminator_{epoch}.m")

    def load(self, model_dir, version):
        super(AdversarialAutoencoder, self).load(model_dir, version)
        self.discriminator = torch_load(
            model_dir / f"discriminator_{version}.m", map_location=get_map_location()
        )

    def initialize_training_components(self):
        super(AdversarialAutoencoder, self).initialize_training_components()
        # define customized optimizers
        self.generator_optimizer = optim.SGD(
            [
                {"params": self.vectorizer.parameters()},
                {"params": self.encoder.parameters()},
                {"params": self.discriminator.parameters()},
            ],
            lr=self._training_settings.generator.lr,
        )
        self.generator_lr_scheduler = CustomLRScheduler(
            self.generator_optimizer,
            factor=self._training_settings.generator.factor,
            patience=self._training_settings.generator.patience,
            min_lr=self._training_settings.generator.min_lr,
        )
        ###
        self.discriminator_optimizer = optim.SGD(
            [{"params": self.discriminator.parameters()}],
            lr=self._training_settings.discriminator.lr,
        )
        self.discriminator_lr_scheduler = CustomLRScheduler(
            self.discriminator_optimizer,
            factor=self._training_settings.discriminator.factor,
            patience=self._training_settings.discriminator.patience,
            min_lr=self._training_settings.discriminator.min_lr,
        )

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
        # train generator
        self.generator_optimizer.zero_grad()
        generator_output = self.forward_generator(one_hot_input)
        generator_loss = self.criterion_NLLLoss(
            generator_output,
            zeros((generator_output.shape[0],), device=device).long(),
        )
        generator_loss.backward()
        self.generator_optimizer.step()
        self.log("generator_loss", generator_loss.item())
        self.log("generator_LR", self.generator_lr_scheduler.get_last_lr())
        self._training_settings.generator.lr = self.generator_lr_scheduler.get_last_lr()
        # train discriminator
        self.discriminator_optimizer.zero_grad()
        ndx = randperm(self.w)
        discriminator_output = self.forward_discriminator(one_hot_input[:, ndx, :])
        discriminator_loss = self.criterion_NLLLoss(
            discriminator_output,
            ones((discriminator_output.shape[0],), device=device).long(),
        )
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        self.log("discriminator_loss", discriminator_loss.item())
        self.log("discriminator_LR", self.discriminator_lr_scheduler.get_last_lr())
        self._training_settings.discriminator.lr = self.discriminator_lr_scheduler.get_last_lr()
        gen_disc_loss = 0.5 * (generator_loss.item() + discriminator_loss.item())
        self.generator_lr_scheduler.step(gen_disc_loss)
        self.discriminator_lr_scheduler.step(gen_disc_loss)
        # clean up
        del input_ndx
        del one_hot_input
        del reconstructor_loss
        del reconstructor_output
        del generator_output
        del generator_loss
        del discriminator_output
        del discriminator_loss

    def test_batch(self, input_vals, device, input_noise=0.0):
        """
        Test a single batch of data, this will move into autoencoder
        :param input_vals:
        :return:
        """
        self.eval()
        with no_grad():
            input_ndx, one_hot_input = self.transform_input(
                input_vals, device, input_noise=input_noise
            )
            (reconstructor_output, generator_output) = self.forward_test(one_hot_input)
            reconstructor_loss = self.criterion_NLLLoss(
                reconstructor_output, input_ndx.reshape((-1,))
            )
            generator_loss = self.criterion_NLLLoss(
                generator_output,
                zeros((generator_output.shape[0],), device=device).long(),
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
            self.log("test_generator_loss", generator_loss.item())
            self.log("test_reconstructor_accuracy", reconstructor_accuracy.item())
            self.log("test_consensus_accuracy", consensus_seq_acc)
            # clean up
            del input_ndx
            del one_hot_input
            del reconstructor_output
            del generator_output
            del reconstructor_loss
            del generator_loss
