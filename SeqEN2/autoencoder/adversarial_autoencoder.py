#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from typing import Dict

from torch import argmax
from torch import load as torch_load
from torch import ones, optim, randperm
from torch import save as torch_save
from torch import sum as torch_sum
from torch import transpose, zeros
from torch.nn.utils import clip_grad_value_

from SeqEN2.autoencoder.autoencoder import Autoencoder
from SeqEN2.autoencoder.utils import CustomLRScheduler, LayerMaker
from SeqEN2.utils.custom_dataclasses import AAETrainingSettings
from SeqEN2.utils.seq_tools import output_to_ndx
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
        return devectorized, discriminator_output, encoded

    def save(self, model_dir, epoch):
        super(AdversarialAutoencoder, self).save(model_dir, epoch)
        torch_save(self.discriminator, model_dir / f"discriminator_{epoch}.m")

    def load(self, model_dir, model_id):
        super(AdversarialAutoencoder, self).load(model_dir, model_id)
        self.discriminator = torch_load(
            model_dir / f"discriminator_{model_id}.m", map_location=get_map_location()
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

    def clip_generator_gradients(self):
        # gradient clipping:
        clip_grad_value_(self.vectorizer.parameters(), clip_value=self.g_clip)
        clip_grad_value_(self.encoder.parameters(), clip_value=self.g_clip)
        clip_grad_value_(self.discriminator.parameters(), clip_value=self.g_clip)

    def clip_discriminator_gradients(self):
        # gradient clipping:
        clip_grad_value_(self.discriminator.parameters(), clip_value=self.g_clip)

    def train_discriminator(self, one_hot_input, device):
        # train generator
        self.generator_optimizer.zero_grad()
        generator_output = self.forward_generator(one_hot_input)
        generator_loss = self.criterion_NLLLoss(
            generator_output,
            zeros((generator_output.shape[0],), device=device).long(),
        )
        generator_loss.backward()
        self.clip_generator_gradients()
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
        self.clip_discriminator_gradients()
        self.discriminator_optimizer.step()
        self.log("discriminator_loss", discriminator_loss.item())
        self.log("discriminator_LR", self.discriminator_lr_scheduler.get_last_lr())
        self._training_settings.discriminator.lr = self.discriminator_lr_scheduler.get_last_lr()
        gen_disc_loss = 0.5 * (generator_loss.item() + discriminator_loss.item())
        self.generator_lr_scheduler.step(gen_disc_loss)
        self.discriminator_lr_scheduler.step(gen_disc_loss)

    def train_one_batch(self, input_vals, input_noise=0.0, device=None, input_keys="A--"):
        if input_vals is not None:
            input_ndx, _, _, one_hot_input = self.transform_input(
                input_vals, device, input_noise=input_noise, input_keys=input_keys
            )
            self.train_reconstructor(one_hot_input, input_ndx)
            # train for continuity
            self.train_continuity(one_hot_input)
            self.train_discriminator(one_hot_input, device)

    def test_discriminator(self, one_hot_input, generator_output, device):
        generator_loss = self.criterion_NLLLoss(
            generator_output,
            zeros((generator_output.shape[0],), device=device).long(),
        )
        generator_choice = argmax(generator_output, dim=1)
        generator_acc = 1 - (torch_sum(generator_choice) / generator_choice.shape[0])

        ndx = randperm(self.w)
        discriminator_output = self.forward_discriminator(one_hot_input[:, ndx, :])
        discriminator_loss = self.criterion_NLLLoss(
            discriminator_output,
            ones((discriminator_output.shape[0],), device=device).long(),
        )
        # discriminator acc
        discriminator_choice = argmax(discriminator_output, dim=1)
        discriminator_acc = torch_sum(discriminator_choice) / discriminator_choice.shape[0]
        # reconstruction_loss, discriminator_loss, classifier_loss
        self.log("test_generator_loss", generator_loss.item())
        self.log("test_generator_accuracy", generator_acc.item())
        self.log("test_discriminator_loss", discriminator_loss.item())
        self.log("test_discriminator_accuracy", discriminator_acc.item())

    def test_one_batch(self, input_vals, device, input_keys="A--"):
        if input_vals is not None:
            input_ndx, _, _, one_hot_input = self.transform_input(
                input_vals, device, input_keys=input_keys
            )
            reconstructor_output, discriminator_output, encoded_output = self.forward_test(
                one_hot_input
            )
            self.test_reconstructor(reconstructor_output, input_ndx, device)
            self.test_discriminator(one_hot_input, discriminator_output, device)
            # test for continuity
            self.test_continuity(encoded_output)

    @staticmethod
    def assert_input_type(input_vals):
        assert isinstance(input_vals, Dict), "AAE requires a dict as input_vals"

    def eval_one_batch(self, input_vals, device, input_keys="A--", embed_only=False):
        if input_vals is not None:
            _, _, _, one_hot_input = self.transform_input(input_vals, device, input_keys=input_keys)
            if embed_only:
                encoded_output = self.forward_embed(one_hot_input)
                return {"embedding": encoded_output}
            else:
                reconstructor_output, _, encoded_output = self.forward_test(one_hot_input)
                return {
                    "reconstructor_output": output_to_ndx(reconstructor_output, self.w),
                    "embedding": encoded_output,
                }
