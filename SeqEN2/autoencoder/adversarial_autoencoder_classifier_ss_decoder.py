#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from typing import Dict

from numpy.random import choice
from torch import Tensor, argmax, cat
from torch import load as torch_load
from torch import no_grad, ones, optim, randperm
from torch import save as torch_save
from torch import sum as torch_sum
from torch import tensor, transpose, zeros

from SeqEN2.autoencoder.adversarial_autoencoder import AdversarialAutoencoder
from SeqEN2.autoencoder.autoencoder_classifier import AutoencoderClassifier
from SeqEN2.autoencoder.autoencoder_ss_decoder import AutoencoderSSDecoder
from SeqEN2.autoencoder.utils import CustomLRScheduler, LayerMaker
from SeqEN2.utils.custom_dataclasses import AAECSSTrainingSettings
from SeqEN2.utils.seq_tools import consensus_acc, get_consensus_seq
from SeqEN2.utils.utils import get_map_location


# class for AAE Classifier
class AdversarialAutoencoderClassifierSSDecoder(
    AdversarialAutoencoder, AutoencoderClassifier, AutoencoderSSDecoder
):
    ds = 9  # SS labels dimension

    def __init__(self, d1, dn, w, arch):
        super(AdversarialAutoencoderClassifierSSDecoder, self).__init__(d1, dn, w, arch)
        # training components
        self._training_settings = AAECSSTrainingSettings()

    @property
    def training_settings(self) -> AAECSSTrainingSettings:
        return self._training_settings

    @training_settings.setter
    def training_settings(self, value=None) -> None:
        if isinstance(value, Dict) or value is None or isinstance(value, AAECSSTrainingSettings):
            if isinstance(value, Dict):
                try:
                    self._training_settings = AAECSSTrainingSettings(**value)
                except TypeError as e:
                    raise KeyError(f"missing/extra keys for AAECSSTrainingSettings, {e}")
            elif isinstance(value, AAECSSTrainingSettings):
                self._training_settings = value
        else:
            raise TypeError(
                f"Training settings must be a dict or None or type AAECSSTrainingSettings, {type(value)} is passed."
            )

    def forward_test(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        decoded = transpose(self.decoder(encoded), 1, 2).reshape(-1, self.d1)
        devectorized = self.devectorizer(decoded)
        discriminator_output = self.discriminator(encoded)
        classifier_output = self.classifier(encoded)
        ss_decoder_output = transpose(self.ss_decoder(encoded), 1, 2).reshape(-1, self.ds)
        return devectorized, discriminator_output, classifier_output, ss_decoder_output, encoded

    def train_one_batch(self, input_vals, input_noise=0.0, device=None, input_keys="A--"):
        if input_vals is not None:
            input_ndx, target_vals_ss, target_vals_cl, one_hot_input = self.transform_input(
                input_vals, device, input_noise=input_noise, input_keys=input_keys
            )
            self.train_reconstructor(one_hot_input, input_ndx)
            # train for continuity
            self.train_continuity(one_hot_input)
            self.train_discriminator(one_hot_input, device)
            if "S" in input_keys:
                self.train_ss_decoder(one_hot_input, target_vals_ss)
            if "C" in input_keys:
                self.test_classifier(one_hot_input, target_vals_cl)

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
                input_vals["ss"], input_noise=input_noise, device=device, input_keys="AS"
            )
        if "clss" in input_vals.keys():
            self.train_one_batch(
                input_vals["clss"], input_noise=input_noise, device=device, input_keys="ACS"
            )

    def test_one_batch(self, input_vals, device, input_keys="A--"):
        if input_vals is not None:
            input_ndx, target_vals_ss, target_vals_cl, one_hot_input = self.transform_input(
                input_vals, device, input_keys=input_keys
            )
            (
                reconstructor_output,
                discriminator_output,
                classifier_output,
                ss_decoder_output,
                encoded_output,
            ) = self.forward_test(one_hot_input)
            self.test_reconstructor(reconstructor_output, input_ndx, device)
            self.train_discriminator(one_hot_input, device)
            # test for continuity
            self.test_continuity(encoded_output)
            if "C" in input_keys:
                self.test_classifier(classifier_output, target_vals_cl)
            if "S" in input_keys:
                self.test_ss_decoder(ss_decoder_output, target_vals_ss, device)

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
                self.test_one_batch(input_vals["ss"], device, input_keys="AS")
            if "clss" in input_vals.keys():
                self.test_one_batch(input_vals["clss"], device, input_keys="ACS")
