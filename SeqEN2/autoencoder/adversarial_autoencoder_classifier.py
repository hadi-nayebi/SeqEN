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

from SeqEN2.autoencoder.adversarial_autoencoder import AdversarialAutoencoder
from SeqEN2.autoencoder.utils import CustomLRScheduler, LayerMaker
from SeqEN2.utils.custom_dataclasses import AAECTrainingSettings
from SeqEN2.utils.seq_tools import consensus_acc
from SeqEN2.utils.utils import get_map_location


# class for AAE Classifier
class AdversarialAutoencoderClassifier(AdversarialAutoencoder):
    def __init__(self, d1, dn, w, arch):
        super(AdversarialAutoencoderClassifier, self).__init__(d1, dn, w, arch)
        assert self.arch.classifier is not None, "arch missing classifier."
        self.classifier = LayerMaker().make(self.arch.classifier)
        # training components
        self._training_settings = AAECTrainingSettings()
        # define customized optimizers
        self.classifier_optimizer = None
        self.classifier_lr_scheduler = None

    @property
    def training_settings(self) -> AAECTrainingSettings:
        return self._training_settings

    @training_settings.setter
    def training_settings(self, value=None) -> None:
        if isinstance(value, Dict) or value is None or isinstance(value, AAECTrainingSettings):
            if isinstance(value, Dict):
                try:
                    self._training_settings = AAECTrainingSettings(**value)
                except TypeError as e:
                    raise KeyError(f"missing/extra keys for AAECTrainingSettings, {e}")
            elif isinstance(value, AAECTrainingSettings):
                self._training_settings = value
        else:
            raise TypeError(
                f"Training settings must be a dict or None or type AAECTrainingSettings, {type(value)} is passed."
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
        discriminator_output = self.discriminator(encoded)
        classifier_output = self.classifier(encoded)
        return devectorized, discriminator_output, classifier_output

    def forward_embed(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        return encoded

    def forward_eval_embed(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        classifier_output = self.classifier(encoded)
        return encoded, classifier_output

    def save(self, model_dir, epoch):
        super(AdversarialAutoencoderClassifier, self).save(model_dir, epoch)
        torch_save(self.classifier, model_dir / f"classifier_{epoch}.m")

    def load(self, model_dir, model_id):
        super(AdversarialAutoencoderClassifier, self).load(model_dir, model_id)
        self.classifier = torch_load(
            model_dir / f"classifier_{model_id}.m", map_location=get_map_location()
        )

    def initialize_training_components(self):
        super(AdversarialAutoencoderClassifier, self).initialize_training_components()
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

    def transform_input(self, input_vals, device, input_noise=0.0):
        raise NotImplementedError("AAEC using custom transform input methods.")

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

    def train_for_classifier(self, one_hot_input, target_vals):
        self.classifier_optimizer.zero_grad()
        classifier_output = self.forward_classifier(one_hot_input)
        classifier_loss = self.criterion_MSELoss(classifier_output, target_vals)
        classifier_loss.backward()
        self.classifier_optimizer.step()
        self.log("classifier_loss", classifier_loss.item())
        self.log("classifier_LR", self.classifier_lr_scheduler.get_last_lr())
        self._training_settings.classifier.lr = self.classifier_lr_scheduler.get_last_lr()
        self.classifier_lr_scheduler.step(classifier_loss.item())

    def train_batch(self, input_vals, device, input_noise=0.0):
        """
        Training for one batch of data, this will move into autoencoder module
        :param input_vals:
        :param device:
        :param input_noise:
        :return:
        """
        self.train()
        input_ndx, target_vals, one_hot_input = self.transform_input_cl(
            input_vals, device, input_noise=input_noise
        )
        # train encoder_decoder
        self.train_for_reconstructor(one_hot_input, input_ndx)
        # train for continuity
        self.train_for_continuity(one_hot_input)
        # train generator and discriminator
        self.train_for_generator_discriminator(one_hot_input, device)
        # train classifier
        self.train_for_classifier(one_hot_input, target_vals)
        # clean up
        del input_ndx
        del one_hot_input

    def test_batch(self, input_vals, device, input_noise=0.0):
        """
        Test a single batch of data, this will move into autoencoder
        :param input_vals:
        :return:
        """
        self.eval()
        with no_grad():
            input_ndx, target_vals, one_hot_input = self.transform_input_cl(
                input_vals, device, input_noise=input_noise
            )
            (
                reconstructor_output,
                generator_output,
                classifier_output,
            ) = self.forward_test(one_hot_input)
            reconstructor_loss = self.criterion_NLLLoss(
                reconstructor_output, input_ndx.reshape((-1,))
            )
            classifier_loss = self.criterion_MSELoss(classifier_output, target_vals)
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
            self.log("test_classifier_loss", classifier_loss.item())
            self.log("test_reconstructor_accuracy", reconstructor_accuracy.item())
            self.log("test_consensus_accuracy", consensus_seq_acc)
            # clean up
            del input_ndx
            del target_vals
            del one_hot_input
            del reconstructor_output
            del generator_output
            del classifier_output
            del reconstructor_loss
            del classifier_loss

    def embed_batch(self, input_vals, device):
        """
        Test a single batch of data, this will move into autoencoder
        :param input_noise:
        :param device:
        :param input_vals:
        :return:
        """
        assert isinstance(input_vals, Tensor), "embed_batch requires a tensor as input_vals"
        self.eval()
        with no_grad():
            # testing with cl data
            input_ndx, target_vals, one_hot_input = self.transform_input_cl(
                input_vals, device, input_noise=0.0
            )
            embedding, classifier_output = self.forward_eval_embed(one_hot_input)
            return embedding, classifier_output
