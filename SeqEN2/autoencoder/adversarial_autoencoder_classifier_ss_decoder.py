#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from typing import Dict

from numpy.random import choice
from torch import Tensor, argmax, cat
from torch import load as torch_load
from torch import no_grad, optim, randperm
from torch import save as torch_save
from torch import sum as torch_sum
from torch import tensor, transpose
from torch.nn.functional import one_hot, unfold

from SeqEN2.autoencoder.adversarial_autoencoder_classifier import (
    AdversarialAutoencoderClassifier,
)
from SeqEN2.autoencoder.utils import CustomLRScheduler, LayerMaker
from SeqEN2.utils.custom_dataclasses import AAECSSTrainingSettings
from SeqEN2.utils.seq_tools import consensus_acc, get_consensus_seq
from SeqEN2.utils.utils import get_map_location


# class for AAE Classifier
class AdversarialAutoencoderClassifierSSDecoder(AdversarialAutoencoderClassifier):

    ds = 9  # SS labels dimension

    def __init__(self, d1, dn, w, arch):
        super(AdversarialAutoencoderClassifierSSDecoder, self).__init__(d1, dn, w, arch)
        assert self.arch.ss_decoder is not None, "arch missing ss_decoder."
        self.ss_decoder = LayerMaker().make(self.arch.ss_decoder)
        # training components
        self._training_settings = AAECSSTrainingSettings()
        # define customized optimizers
        self.ss_decoder_optimizer = None
        self.ss_decoder_lr_scheduler = None

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

    def forward(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        decoded = transpose(self.decoder(encoded), 1, 2).reshape(-1, self.d1)
        devectorized = self.devectorizer(decoded)
        generator_output = self.discriminator(encoded)
        classifier_output = self.classifier(encoded)
        ss_decoder_output = transpose(self.ss_decoder(encoded), 1, 2).reshape(-1, self.ds)
        return devectorized, generator_output, classifier_output, ss_decoder_output, encoded

    def forward_embed(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        classifier_output = self.classifier(encoded)
        ss_decoder_output = transpose(self.ss_decoder(encoded), 1, 2).reshape(-1, self.ds)
        return encoded, classifier_output, ss_decoder_output

    def save(self, model_dir, epoch):
        super(AdversarialAutoencoderClassifierSSDecoder, self).save(model_dir, epoch)
        torch_save(self.ss_decoder, model_dir / f"ss_decoder_{epoch}.m")

    def load(self, model_dir, model_id):
        super(AdversarialAutoencoderClassifierSSDecoder, self).load(model_dir, model_id)
        self.ss_decoder = torch_load(
            model_dir / f"ss_decoder_{model_id}.m", map_location=get_map_location()
        )

    def initialize_training_components(self):
        super(AdversarialAutoencoderClassifierSSDecoder, self).initialize_training_components()
        # define customized optimizers
        self.ss_decoder_optimizer = optim.SGD(
            [
                {"params": self.vectorizer.parameters()},
                {"params": self.encoder.parameters()},
                {"params": self.ss_decoder.parameters()},
            ],
            lr=self._training_settings.ss_decoder.lr,
        )
        self.ss_decoder_lr_scheduler = CustomLRScheduler(
            self.ss_decoder_optimizer,
            factor=self._training_settings.ss_decoder.factor,
            patience=self._training_settings.ss_decoder.patience,
            min_lr=self._training_settings.ss_decoder.min_lr,
        )

    def transform_input(self, input_vals, device, input_noise=0.0):
        raise NotImplementedError("AAECSS using custom transform input methods.")

    def transform_input_cl(self, input_vals, device, input_noise=0.0):
        # scans by sliding window of w
        assert isinstance(input_vals, Tensor)
        kernel_size = (input_vals.shape[1], self.w)
        input_vals = unfold(input_vals.T[None, None, :, :], kernel_size=kernel_size)[0].T
        input_ndx = input_vals[:, : self.w].long()
        target_vals = input_vals[:, self.w :].mean(axis=1).reshape((-1, 1))
        target_vals = cat((target_vals, 1 - target_vals), 1).float()
        one_hot_input = one_hot(input_ndx, num_classes=self.d0) * 1.0
        if input_noise > 0.0:
            ndx = randperm(self.w)
            size = list(one_hot_input.shape)
            size[-1] = 1
            p = tensor(choice([1, 0], p=[input_noise, 1 - input_noise], size=size)).to(device)
            mutated_one_hot = (one_hot_input[:, ndx, :] * p) + (one_hot_input * (1 - p))
            return input_ndx, target_vals, mutated_one_hot
        else:
            return input_ndx, target_vals, one_hot_input

    def transform_input_ss(self, input_vals, device, input_noise=0.0):
        # scans by sliding window of w
        assert isinstance(input_vals, Tensor)
        kernel_size = (input_vals.shape[1], self.w)
        input_vals = unfold(input_vals.float().T[None, None, :, :], kernel_size=kernel_size)[0].T
        input_ndx = input_vals[:, : self.w].long()
        target_vals = input_vals[:, self.w :].long()
        one_hot_input = one_hot(input_ndx, num_classes=self.d0) * 1.0
        if input_noise > 0.0:
            ndx = randperm(self.w)
            size = list(one_hot_input.shape)
            size[-1] = 1
            p = tensor(choice([1, 0], p=[input_noise, 1 - input_noise], size=size)).to(device)
            mutated_one_hot = (one_hot_input[:, ndx, :] * p) + (one_hot_input * (1 - p))
            return input_ndx, target_vals, mutated_one_hot
        else:
            return input_ndx, target_vals, one_hot_input

    def transform_input_clss(self, input_vals, device, input_noise=0.0):
        # scans by sliding window of w
        assert isinstance(input_vals, Tensor)
        kernel_size = (input_vals.shape[1], self.w)
        input_vals = unfold(input_vals.float().T[None, None, :, :], kernel_size=kernel_size)[0].T
        input_ndx = input_vals[:, : self.w].long()
        target_cl = input_vals[:, self.w : -self.w].mean(axis=1).reshape((-1, 1))
        target_ss = input_vals[:, -self.w :].long()
        one_hot_input = one_hot(input_ndx, num_classes=self.d0) * 1.0
        if input_noise > 0.0:
            ndx = randperm(self.w)
            size = list(one_hot_input.shape)
            size[-1] = 1
            p = tensor(choice([1, 0], p=[input_noise, 1 - input_noise], size=size)).to(device)
            mutated_one_hot = (one_hot_input[:, ndx, :] * p) + (one_hot_input * (1 - p))
            return input_ndx, target_cl, target_ss, mutated_one_hot
        else:
            return input_ndx, target_cl, target_ss, one_hot_input

    def train_for_ss_decoder(self, ss_decoder_output, target_vals):
        self.ss_decoder_optimizer.zero_grad()
        ss_decoder_loss = self.criterion_NLLLoss(ss_decoder_output, target_vals.reshape((-1,)))
        ss_decoder_loss.backward()
        self.ss_decoder_optimizer.step()
        self._training_settings.ss_decoder.lr = self.ss_decoder_lr_scheduler.get_last_lr()
        self.ss_decoder_lr_scheduler.step(ss_decoder_loss.item())
        self.log("ss_decoder_loss", ss_decoder_loss.item())
        self.log("ss_decoder_LR", self.training_settings.ss_decoder.lr)

    def train_batch(self, input_vals, device, input_noise=0.0):
        """
        Training for one batch of data, this will move into autoencoder module
        :param input_vals:
        :param device:
        :param input_noise:
        :return:
        """
        assert isinstance(input_vals, Dict), "AAECSS requires a dict as input_vals"
        key_check = "ss" in input_vals.keys() and "cl" in input_vals.keys()
        assert key_check, "input_vals must contain cl and ss keys"
        self.train()
        # training with cl data
        input_ndx, target_vals, one_hot_input = self.transform_input_cl(
            input_vals["cl"], device, input_noise=input_noise
        )
        # forward
        (
            reconstructor_output,
            generator_output,
            classifier_output,
            ss_decoder_output,
            encoded_output,
        ) = self.forward(one_hot_input)
        # train encoder_decoder
        self.train_for_reconstructor(reconstructor_output, input_ndx)
        # train for continuity
        self.train_for_continuity(encoded_output)
        # train generator and discriminator
        self.train_for_generator_discriminator(generator_output, one_hot_input, device)
        # train classifier
        self.train_for_classifier(classifier_output, target_vals)
        # training with ss data
        input_ndx, target_vals, one_hot_input = self.transform_input_ss(
            input_vals["ss"], device, input_noise=input_noise
        )
        # forward
        (
            reconstructor_output,
            generator_output,
            classifier_output,
            ss_decoder_output,
            encoded_output,
        ) = self.forward(one_hot_input)
        # train encoder_decoder
        self.train_for_reconstructor(reconstructor_output, input_ndx)
        # train for continuity
        self.train_for_continuity(encoded_output)
        # train generator and discriminator
        self.train_for_generator_discriminator(generator_output, one_hot_input, device)
        # train encoder_ss_decoder
        self.train_for_ss_decoder(ss_decoder_output, target_vals)
        # training with clss data
        if "clss" in input_vals.keys():
            input_ndx, target_cl, target_ss, one_hot_input = self.transform_input_ss(
                input_vals["clss"], device, input_noise=input_noise
            )
            # forward
            (
                reconstructor_output,
                generator_output,
                classifier_output,
                ss_decoder_output,
                encoded_output,
            ) = self.forward(one_hot_input)
            # train encoder_decoder
            self.train_for_reconstructor(reconstructor_output, input_ndx)
            # train for continuity
            self.train_for_continuity(encoded_output)
            # train generator and discriminator
            self.train_for_generator_discriminator(generator_output, one_hot_input, device)
            # train classifier
            self.train_for_classifier(classifier_output, target_vals)
            # train encoder_ss_decoder
            self.train_for_ss_decoder(ss_decoder_output, target_vals)

    def test_for_ss_decoder(self, ss_decoder_output, target_vals, device):
        ss_decoder_loss = self.criterion_NLLLoss(ss_decoder_output, target_vals.reshape((-1,)))
        self.log("test_ss_decoder_loss", ss_decoder_loss.item())
        # ss_decoder acc
        ss_decoder_ndx = argmax(ss_decoder_output, dim=1)
        ss_decoder_accuracy = (
            torch_sum(ss_decoder_ndx == target_vals.reshape((-1,))) / ss_decoder_ndx.shape[0]
        )
        consensus_ss_acc, _ = consensus_acc(
            target_vals, ss_decoder_ndx.reshape((-1, self.w)), device
        )
        self.log("test_ss_decoder_accuracy", ss_decoder_accuracy.item())
        self.log("test_consensus_ss_accuracy", consensus_ss_acc)

    def test_batch(self, input_vals, device, input_noise=0.0):
        """
        Test a single batch of data, this will move into autoencoder
        :param input_vals:
        :return:
        """
        assert isinstance(input_vals, Dict), "AAECSS requires a dict as input_vals"
        key_check = "ss" in input_vals.keys() and "cl" in input_vals.keys()
        assert key_check, "input_vals must contain cl and ss keys"
        self.eval()
        with no_grad():
            # testing with cl data
            input_ndx, target_vals, one_hot_input = self.transform_input_cl(
                input_vals["cl"], device
            )
            # test
            (
                reconstructor_output,
                generator_output,
                classifier_output,
                _,
                encoded_output,
            ) = self.forward(one_hot_input)
            # test for constructor
            self.test_for_constructor(reconstructor_output, input_ndx, device)
            # test continuity loss
            self.test_for_continuity(encoded_output)
            # test generator and discriminator
            self.test_for_generator_discriminator(one_hot_input, generator_output, device)
            # test for classifier
            self.test_for_classifier(classifier_output, target_vals)
            # clean up
            del input_ndx
            del target_vals
            del one_hot_input
            # testing with ss data
            input_ndx, target_vals, one_hot_input = self.transform_input_ss(
                input_vals["ss"], device
            )
            # test
            (
                reconstructor_output,
                generator_output,
                _,
                ss_decoder_output,
                encoded_output,
            ) = self.forward(one_hot_input)
            # test for constructor
            self.test_for_constructor(reconstructor_output, input_ndx, device)
            # test continuity loss
            self.test_for_continuity(encoded_output)
            # test generator and discriminator
            self.test_for_generator_discriminator(one_hot_input, generator_output, device)
            # test for ss_decoder
            self.test_for_ss_decoder(ss_decoder_output, target_vals, device)
            # clean up
            del input_ndx
            del target_vals
            del one_hot_input
            # training with clss data
            if "clss" in input_vals.keys():
                input_ndx, target_cl, target_ss, one_hot_input = self.transform_input_ss(
                    input_vals["clss"], device
                )
                # test
                (
                    reconstructor_output,
                    generator_output,
                    classifier_output,
                    ss_decoder_output,
                    encoded_output,
                ) = self.forward(one_hot_input)
                # test for constructor
                self.test_for_constructor(reconstructor_output, input_ndx, device)
                # test continuity loss
                self.test_for_continuity(encoded_output)
                # test generator and discriminator
                self.test_for_generator_discriminator(one_hot_input, generator_output, device)
                # test for classifier
                self.test_for_classifier(classifier_output, target_vals)
                # test for ss_decoder
                self.test_for_ss_decoder(ss_decoder_output, target_vals, device)

    def embed_batch(self, input_vals, device):
        """
        Test a single batch of data, this will move into autoencoder
        :param device:
        :param input_vals:
        :return:
        """
        assert isinstance(input_vals, Tensor), "embed_batch requires a tensor as input_vals"
        self.eval()
        with no_grad():
            # testing with cl data
            input_ndx, target_vals, one_hot_input = self.transform_input_cl(input_vals, device)
            (
                embedding,
                classifier_output,
                ss_decoder_output,
            ) = self.forward_embed(one_hot_input)
            consensus_ss = get_consensus_seq(
                argmax(ss_decoder_output, dim=1).reshape((-1, self.w)), device
            )
            return embedding, classifier_output, consensus_ss
