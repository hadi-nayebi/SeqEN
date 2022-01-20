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
from torch.nn.functional import one_hot, unfold

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
        assert self.arch.ss_decoder is not None, "arch missing ss_decoder."
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
        ss_decoder_output = transpose(self.ss_decoder(encoded), 1, 2).reshape(-1, self.ds)
        return devectorized, discriminator_output, classifier_output, ss_decoder_output

    def save(self, model_dir, epoch):
        super(AdversarialAutoencoderClassifierSSDecoder, self).save(model_dir, epoch)
        torch_save(self.ss_decoder, model_dir / f"ss_decoder_{epoch}.m")

    def load(self, model_dir, version):
        super(AdversarialAutoencoderClassifierSSDecoder, self).load(model_dir, version)
        self.ss_decoder = torch_load(
            model_dir / f"ss_decoder_{version}.m", map_location=get_map_location()
        )

    def set_training_params(self, training_params=None):
        if training_params is None:
            self.training_params = {
                key: {"lr": 0.01, "factor": 0.9, "patience": 10000, "min_lr": 0.00001}
                for key in [
                    "reconstructor",
                    "generator",
                    "discriminator",
                    "classifier",
                    "ss_decoder",
                ]
            }
            self.training_params["gen"] = 3
        else:
            assert (
                self.training_params["gen"] > 2
            ), "Training params is from older gen, require 3 or larger"
            assert (
                "ss_decoder" in self.training_params.keys()
            ), "ss_decoder is missing in training params."
            self.training_params = training_params

    def initialize_training_components(self):
        super(AdversarialAutoencoderClassifierSSDecoder, self).initialize_training_components()
        # define customized optimizers
        self.ss_decoder_optimizer = optim.SGD(
            [
                {"params": self.vectorizer.parameters()},
                {"params": self.encoder.parameters()},
                {"params": self.ss_decoder.parameters()},
            ],
            lr=self.training_params["ss_decoder"]["lr"],
        )
        self.ss_decoder_lr_scheduler = CustomLRScheduler(
            self.ss_decoder_optimizer,
            factor=self.training_params["ss_decoder"]["factor"],
            patience=self.training_params["ss_decoder"]["patience"],
            min_lr=self.training_params["ss_decoder"]["min_lr"],
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

    def train_batch(self, input_vals, device, input_noise=0.0, wandb_log=True):
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
        # train encoder_decoder
        self.reconstructor_optimizer.zero_grad()
        reconstructor_output = self.forward_encoder_decoder(one_hot_input)
        reconstructor_loss = self.criterion_NLLLoss(reconstructor_output, input_ndx.reshape((-1,)))
        reconstructor_loss.backward()
        self.reconstructor_optimizer.step()
        if wandb_log:
            wandb.log({"reconstructor_loss_cl": reconstructor_loss.item()})
            wandb.log({"reconstructor_LR": self.reconstructor_lr_scheduler.get_last_lr()})
        self.training_params["reconstructor"]["lr"] = self.reconstructor_lr_scheduler.get_last_lr()
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
        if wandb_log:
            wandb.log({"generator_loss_cl": generator_loss.item()})
            wandb.log({"generator_LR": self.generator_lr_scheduler.get_last_lr()})
        self.training_params["generator"]["lr"] = self.generator_lr_scheduler.get_last_lr()
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
        if wandb_log:
            wandb.log({"discriminator_loss_cl": discriminator_loss.item()})
            wandb.log({"discriminator_LR": self.discriminator_lr_scheduler.get_last_lr()})
        self.training_params["discriminator"]["lr"] = self.discriminator_lr_scheduler.get_last_lr()
        gen_disc_loss = 0.5 * (generator_loss.item() + discriminator_loss.item())
        self.generator_lr_scheduler.step(gen_disc_loss)
        self.discriminator_lr_scheduler.step(gen_disc_loss)
        # train classifier
        self.classifier_optimizer.zero_grad()
        classifier_output = self.forward_classifier(one_hot_input)
        classifier_loss = self.criterion_MSELoss(classifier_output, target_vals)
        classifier_loss.backward()
        self.classifier_optimizer.step()
        if wandb_log:
            wandb.log({"classifier_loss": classifier_loss.item()})
            wandb.log({"classifier_LR": self.classifier_lr_scheduler.get_last_lr()})
        self.training_params["classifier"]["lr"] = self.classifier_lr_scheduler.get_last_lr()
        self.classifier_lr_scheduler.step(classifier_loss.item())
        # clean up
        del reconstructor_loss
        del reconstructor_output
        del generator_output
        del generator_loss
        del discriminator_output
        del discriminator_loss
        del classifier_output
        del classifier_loss
        # training with ss data
        input_ndx, target_vals, one_hot_input = self.transform_input_ss(
            input_vals["ss"], device, input_noise=input_noise
        )
        # train encoder_decoder
        self.reconstructor_optimizer.zero_grad()
        reconstructor_output = self.forward_encoder_decoder(one_hot_input)
        reconstructor_loss = self.criterion_NLLLoss(reconstructor_output, input_ndx.reshape((-1,)))
        reconstructor_loss.backward()
        self.reconstructor_optimizer.step()
        if wandb_log:
            wandb.log({"reconstructor_loss_ss": reconstructor_loss.item()})
            wandb.log({"reconstructor_LR": self.reconstructor_lr_scheduler.get_last_lr()})
        self.training_params["reconstructor"]["lr"] = self.reconstructor_lr_scheduler.get_last_lr()
        self.reconstructor_lr_scheduler.step(reconstructor_loss.item())
        # train encoder_SS_decoder
        self.ss_decoder_optimizer.zero_grad()
        ss_decoder_output = self.forward_ss_decoder(one_hot_input)
        ss_decoder_loss = self.criterion_NLLLoss(ss_decoder_output, target_vals.reshape((-1,)))
        ss_decoder_loss.backward()
        self.ss_decoder_optimizer.step()
        if wandb_log:
            wandb.log({"ss_decoder_loss": ss_decoder_loss.item()})
            wandb.log({"ss_decoder_LR": self.ss_decoder_lr_scheduler.get_last_lr()})
        self.training_params["ss_decoder"]["lr"] = self.ss_decoder_lr_scheduler.get_last_lr()
        self.ss_decoder_lr_scheduler.step(ss_decoder_loss.item())
        # train generator
        self.generator_optimizer.zero_grad()
        generator_output = self.forward_generator(one_hot_input)
        generator_loss = self.criterion_NLLLoss(
            generator_output,
            zeros((generator_output.shape[0],), device=device).long(),
        )
        generator_loss.backward()
        self.generator_optimizer.step()
        if wandb_log:
            wandb.log({"generator_loss_ss": generator_loss.item()})
            wandb.log({"generator_LR": self.generator_lr_scheduler.get_last_lr()})
        self.training_params["generator"]["lr"] = self.generator_lr_scheduler.get_last_lr()
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
        if wandb_log:
            wandb.log({"discriminator_loss_ss": discriminator_loss.item()})
            wandb.log({"discriminator_LR": self.discriminator_lr_scheduler.get_last_lr()})
        self.training_params["discriminator"]["lr"] = self.discriminator_lr_scheduler.get_last_lr()
        gen_disc_loss = 0.5 * (generator_loss.item() + discriminator_loss.item())
        self.generator_lr_scheduler.step(gen_disc_loss)
        self.discriminator_lr_scheduler.step(gen_disc_loss)
        # clean up
        del reconstructor_loss
        del reconstructor_output
        del generator_output
        del generator_loss
        del discriminator_output
        del discriminator_loss
        del ss_decoder_output
        del ss_decoder_loss

    def test_batch(self, input_vals, device, input_noise=0.0, wandb_log=True):
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
                input_vals["cl"], device, input_noise=input_noise
            )
            (
                reconstructor_output,
                generator_output,
                classifier_output,
                ss_decoder_output,
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
            if wandb_log:
                wandb.log({"test_reconstructor_loss_cl": reconstructor_loss.item()})
                wandb.log({"test_generator_loss_cl": generator_loss.item()})
                wandb.log({"test_classifier_loss": classifier_loss.item()})
                wandb.log({"test_reconstructor_accuracy_cl": reconstructor_accuracy.item()})
                wandb.log({"test_consensus_accuracy_cl": consensus_seq_acc})
            # clean up
            del reconstructor_output
            del generator_output
            del classifier_output
            del reconstructor_loss
            del generator_loss
            del target_vals
            del classifier_loss
            # testing with ss data
            input_ndx, target_vals, one_hot_input = self.transform_input_ss(
                input_vals["ss"], device, input_noise=input_noise
            )
            (
                reconstructor_output,
                generator_output,
                classifier_output,
                ss_decoder_output,
            ) = self.forward_test(one_hot_input)
            reconstructor_loss = self.criterion_NLLLoss(
                reconstructor_output, input_ndx.reshape((-1,))
            )
            generator_loss = self.criterion_NLLLoss(
                generator_output,
                zeros((generator_output.shape[0],), device=device).long(),
            )
            ss_decoder_loss = self.criterion_NLLLoss(ss_decoder_output, target_vals)
            # reconstructor acc
            reconstructor_ndx = argmax(reconstructor_output, dim=1)
            reconstructor_accuracy = (
                torch_sum(reconstructor_ndx == input_ndx.reshape((-1,)))
                / reconstructor_ndx.shape[0]
            )
            consensus_seq_acc, _ = consensus_acc(
                input_ndx, reconstructor_ndx.reshape((-1, self.w)), device
            )
            # ss_decoder acc
            ss_decoder_ndx = argmax(ss_decoder_output, dim=1)
            ss_decoder_accuracy = (
                torch_sum(ss_decoder_ndx == target_vals.reshape((-1,))) / ss_decoder_ndx.shape[0]
            )
            consensus_ss_acc, _ = consensus_acc(
                target_vals, ss_decoder_ndx.reshape((-1, self.w)), device
            )
            # reconstruction_loss, discriminator_loss, classifier_loss
            if wandb_log:
                wandb.log({"test_reconstructor_loss_ss": reconstructor_loss.item()})
                wandb.log({"test_generator_loss_ss": generator_loss.item()})
                wandb.log({"test_ss_decoder_loss": ss_decoder_loss.item()})
                wandb.log({"test_reconstructor_accuracy_ss": reconstructor_accuracy.item()})
                wandb.log({"test_consensus_accuracy_ss": consensus_seq_acc})
                wandb.log({"test_ss_decoder_accuracy": ss_decoder_accuracy.item()})
                wandb.log({"test_consensus_ss_accuracy": consensus_ss_acc})
            # clean up
            del reconstructor_output
            del generator_output
            del classifier_output
            del reconstructor_loss
            del generator_loss
            del target_vals
            del ss_decoder_loss
