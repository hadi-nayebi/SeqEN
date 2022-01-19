#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


from torch import argmax
from torch import load as torch_load
from torch import no_grad, optim
from torch import save as torch_save
from torch import sum as torch_sum
from torch import transpose, zeros
from torch.nn import MSELoss

import wandb
from SeqEN2.autoencoder.adversarial_autoencoder_classifier import (
    AdversarialAutoencoderClassifier,
)
from SeqEN2.autoencoder.utils import CustomLRScheduler, LayerMaker
from SeqEN2.utils.seq_tools import consensus_acc
from SeqEN2.utils.utils import get_map_location


# class for AAE Classifier
class AdversarialAutoencoderClassifierSSDecoder(AdversarialAutoencoderClassifier):

    ds = 9  # SS labels dimension

    def __init__(self, d1, dn, w, arch):
        super(AdversarialAutoencoderClassifierSSDecoder, self).__init__(d1, dn, w, arch)
        self.ss_decoder = LayerMaker().make(self.arch.ss_decoder)
        # define customized optimizers
        self.ss_decoder_optimizer = None
        self.ss_decoder_lr_scheduler = None

    def forward_ss_decoder(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        ss_decoder_output = transpose(self.ss_decoder(encoded), 1, 2).reshape(-1, self.ds)
        return ss_decoder_output

    def forward_test(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        decoded = transpose(self.decoder(encoded), 1, 2).reshape(-1, self.d1)
        devectorized = self.devectorizer(decoded)
        discriminator_output = self.discriminator(encoded)
        classifier_output = self.classifier(encoded)
        return devectorized, discriminator_output, classifier_output

    def save(self, model_dir, epoch):
        super(AdversarialAutoencoderClassifier, self).save(model_dir, epoch)
        torch_save(self.classifier, model_dir / f"classifier_{epoch}.m")

    def load(self, model_dir, version):
        super(AdversarialAutoencoderClassifier, self).load(model_dir, version)
        self.classifier = torch_load(
            model_dir / f"classifier_{version}.m", map_location=get_map_location()
        )

    def set_training_params(self, training_params=None):
        if training_params is None:
            self.training_params = {
                key: {"lr": 0.01, "factor": 0.9, "patience": 10000, "min_lr": 0.00001}
                for key in ["reconstructor", "generator", "discriminator", "classifier"]
            }
        else:
            self.training_params = training_params

    def initialize_training_components(self):
        super(AdversarialAutoencoderClassifier, self).initialize_training_components()
        # define customized optimizers
        self.classifier_optimizer = optim.SGD(
            [
                {"params": self.vectorizer.parameters()},
                {"params": self.encoder.parameters()},
                {"params": self.classifier.parameters()},
            ],
            lr=self.training_params["classifier"]["lr"],
        )
        self.classifier_lr_scheduler = CustomLRScheduler(
            self.classifier_optimizer,
            factor=self.training_params["classifier"]["factor"],
            patience=self.training_params["classifier"]["patience"],
            min_lr=self.training_params["classifier"]["min_lr"],
        )

    def train_batch(self, input_vals, device, input_noise=0.0):
        """
        Training for one batch of data, this will move into autoencoder module
        :param input_vals:
        :param device:
        :param input_noise:
        :return:
        """
        super(AdversarialAutoencoderClassifier, self).train_batch(
            input_vals, device, input_noise=input_noise
        )
        # train classifier
        self.classifier_optimizer.zero_grad()
        classifier_output = self.forward_classifier(self.one_hot_input)
        classifier_loss = self.criterion_MSELoss(classifier_output, self.target_vals)
        classifier_loss.backward()
        self.classifier_optimizer.step()
        wandb.log({"classifier_loss": classifier_loss.item()})
        wandb.log({"classifier_LR": self.classifier_lr_scheduler.get_last_lr()})
        self.training_params["classifier"]["lr"] = self.classifier_lr_scheduler.get_last_lr()
        self.classifier_lr_scheduler.step(classifier_loss.item())
        # clean up
        del classifier_output
        del classifier_loss

    def test_batch(self, input_vals, device, input_noise=0.0):
        """
        Test a single batch of data, this will move into autoencoder
        :param input_vals:
        :return:
        """
        self.eval()
        with no_grad():
            input_ndx, target_vals, one_hot_input = self.transform_input(
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
            generator_loss = self.criterion_NLLLoss(
                generator_output,
                zeros((generator_output.shape[0],), device=device).long(),
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
            wandb.log({"test_reconstructor_loss": reconstructor_loss.item()})
            wandb.log({"test_generator_loss": generator_loss.item()})
            wandb.log({"test_classifier_loss": classifier_loss.item()})
            wandb.log({"test_reconstructor_accuracy": reconstructor_accuracy.item()})
            wandb.log({"test_consensus_accuracy": consensus_seq_acc})
            # clean up
            del reconstructor_output
            del generator_output
            del classifier_output
            del reconstructor_loss
            del generator_loss
            del target_vals
            del classifier_loss
