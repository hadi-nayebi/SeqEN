#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from typing import Dict

from torch import argmax
from torch import load as torch_load
from torch import mul, no_grad, optim
from torch import save as torch_save
from torch import sum as torch_sum
from torch import transpose
from torch.nn.utils import clip_grad_value_

from SeqEN2.autoencoder.autoencoder import Autoencoder
from SeqEN2.autoencoder.utils import CustomLRScheduler, LayerMaker
from SeqEN2.utils.custom_dataclasses import AECTrainingSettings
from SeqEN2.utils.seq_tools import get_consensus_seq
from SeqEN2.utils.utils import get_map_location


# class for AAE
class AutoencoderClassifier(Autoencoder):
    def __init__(self, d1, dn, w, arch):
        super(AutoencoderClassifier, self).__init__(d1, dn, w, arch)
        assert self.arch.classifier is not None, "arch missing classifier."
        self.classifier = LayerMaker().make(self.arch.classifier)
        # training components
        self._training_settings = AECTrainingSettings()
        ###
        self.classifier_optimizer = None
        self.classifier_lr_scheduler = None

    @property
    def training_settings(self) -> AECTrainingSettings:
        return self._training_settings

    @training_settings.setter
    def training_settings(self, value=None) -> None:
        if isinstance(value, Dict) or value is None or isinstance(value, AECTrainingSettings):
            if isinstance(value, Dict):
                try:
                    self._training_settings = AECTrainingSettings(**value)
                except TypeError as e:
                    raise KeyError(f"missing/extra keys for AECTrainingSettings, {e}")
            elif isinstance(value, AECTrainingSettings):
                self._training_settings = value
        else:
            raise TypeError(
                f"Training settings must be a dict or None or type AECTrainingSettings, {type(value)} is passed."
            )

    def forward_classifier(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        classifier_output = self.classifier(encoded)
        return classifier_output

    def forward_test(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        decoded = transpose(self.decoder(encoded), 1, 2).reshape(-1, self.d1)
        devectorized = self.devectorizer(decoded)
        classifier_output = self.classifier(encoded)
        return devectorized, classifier_output, encoded

    def save(self, model_dir, epoch):
        super(AutoencoderClassifier, self).save(model_dir, epoch)
        torch_save(self.classifier, model_dir / f"classifier_{epoch}.m")

    def load(self, model_dir, model_id):
        super(AutoencoderClassifier, self).load(model_dir, model_id)
        self.classifier = torch_load(
            model_dir / f"classifier_{model_id}.m", map_location=get_map_location()
        )

    def initialize_training_components(self):
        super(AutoencoderClassifier, self).initialize_training_components()
        # define customized optimizers
        self.classifier_optimizer = optim.SGD(
            [
                {"params": self.vectorizer.parameters()},
                {"params": self.encoder.parameters()},
                {"params": self.classifier.parameters()},
            ],
            lr=self._training_settings.classifier.lr,
        )
        self.classifier_lr_scheduler = CustomLRScheduler(
            self.classifier_optimizer,
            factor=self._training_settings.classifier.factor,
            patience=self._training_settings.classifier.patience,
            min_lr=self._training_settings.classifier.min_lr,
        )

    def clip_classifier_gradients(self):
        # gradient clipping:
        clip_grad_value_(self.vectorizer.parameters(), clip_value=self.g_clip)
        clip_grad_value_(self.encoder.parameters(), clip_value=self.g_clip)
        clip_grad_value_(self.classifier.parameters(), clip_value=self.g_clip)

    def train_classifier(self, one_hot_input, target_vals_cl):
        # train classifier
        self.classifier_optimizer.zero_grad()
        classifier_output = self.forward_classifier(one_hot_input)
        classifier_loss = self.criterion_MSELoss(classifier_output, target_vals_cl)
        classifier_loss.backward()
        self.clip_classifier_gradients()
        self.classifier_optimizer.step()
        self.log("classifier_loss", classifier_loss.item())
        self.log("classifier_LR", self.classifier_lr_scheduler.get_last_lr())
        self._training_settings.classifier.lr = self.classifier_lr_scheduler.get_last_lr()
        self.classifier_lr_scheduler.step(classifier_loss.item())

    def train_one_batch(self, input_vals, input_noise=0.0, device=None, input_keys="A-C"):
        if input_vals is not None:
            input_ndx, _, target_vals_cl, one_hot_input = self.transform_input(
                input_vals, device, input_noise=input_noise, input_keys=input_keys
            )
            self.train_reconstructor(one_hot_input, input_ndx)
            # train for continuity
            self.train_continuity(one_hot_input)
            if "C" in input_keys:
                self.train_classifier(one_hot_input, target_vals_cl)

    @staticmethod
    def assert_input_type(input_vals):
        assert isinstance(input_vals, Dict), "AEC requires a dict as input_vals"

    def train_batch(self, input_vals, device, input_noise=0.0):
        """
        Training for one batch of data, this will move into autoencoder module
        :param input_vals:
        :param device:
        :param input_noise:
        :return:
        """
        self.assert_input_type(input_vals)
        self.train()
        if "cl" in input_vals.keys():
            self.train_one_batch(
                input_vals["cl"], input_noise=input_noise, device=device, input_keys="AC"
            )
        if "ss" in input_vals.keys():
            self.train_one_batch(
                input_vals["ss"], input_noise=input_noise, device=device, input_keys="A-"
            )
        if "clss" in input_vals.keys():
            self.train_one_batch(
                input_vals["clss"], input_noise=input_noise, device=device, input_keys="A-C"
            )

    def test_classifier(self, classifier_output, target_vals_cl):
        classifier_loss = self.criterion_MSELoss(classifier_output, target_vals_cl)
        classifier_choice = argmax(classifier_output, dim=1)
        classifier_target = argmax(target_vals_cl, dim=1)
        classifier_acc_p = torch_sum(mul(classifier_choice, classifier_target)) / torch_sum(
            classifier_target
        )
        classifier_acc_n = torch_sum(mul(1 - classifier_choice, 1 - classifier_target)) / torch_sum(
            1 - classifier_target
        )
        self.log("test_classifier_acc_p", classifier_acc_p.item())
        self.log("test_classifier_acc_n", classifier_acc_n.item())
        self.log("test_classifier_loss", classifier_loss.item())

    def test_one_batch(self, input_vals, device, input_keys="A-C"):
        if input_vals is not None:
            input_ndx, _, target_vals_cl, one_hot_input = self.transform_input(
                input_vals, device, input_keys=input_keys
            )
            reconstructor_output, classifier_output, encoded_output = self.forward_test(
                one_hot_input
            )
            self.test_reconstructor(reconstructor_output, input_ndx, device)
            # test for continuity
            self.test_continuity(encoded_output)
            if "C" in input_keys:
                self.test_classifier(classifier_output, target_vals_cl)

    def test_batch(self, input_vals, device):
        """
        Test a single batch of data, this will move into autoencoder
        :param input_vals:
        :return:
        """
        self.assert_input_type(input_vals)
        self.eval()
        with no_grad():
            if "cl" in input_vals.keys():
                self.test_one_batch(input_vals["cl"], device, input_keys="AC")
            if "ss" in input_vals.keys():
                self.test_one_batch(input_vals["ss"], device, input_keys="A-")
            if "clss" in input_vals.keys():
                self.test_one_batch(input_vals["clss"], device, input_keys="A-C")

    def eval_one_batch(self, input_vals, device, input_keys="A--", embed_only=False):
        if input_vals is not None:
            _, _, _, one_hot_input = self.transform_input(input_vals, device, input_keys=input_keys)
            if embed_only:
                encoded_output = self.forward_embed(one_hot_input)
                return {"embedding": encoded_output}
            else:
                reconstructor_output, classifier_output, encoded_output = self.forward_test(
                    one_hot_input
                )
                return {
                    "reconstructor_output": reconstructor_output,
                    "classifier_output": classifier_output,
                    "embedding": encoded_output,
                }
