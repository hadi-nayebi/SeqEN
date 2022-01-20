#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from numpy.random import choice
from torch import Tensor, argmax, cat
from torch import load as torch_load
from torch import no_grad, ones, optim, randperm
from torch import save as torch_save
from torch import sum as torch_sum
from torch import tensor, transpose, zeros
from torch.nn.functional import one_hot, unfold

import wandb
from SeqEN2.autoencoder.autoencoder import Autoencoder
from SeqEN2.autoencoder.utils import CustomLRScheduler, LayerMaker
from SeqEN2.utils.seq_tools import consensus_acc
from SeqEN2.utils.utils import get_map_location


# class for AAE
class AdversarialAutoencoder(Autoencoder):
    def __init__(self, d1, dn, w, arch):
        super(AdversarialAutoencoder, self).__init__(d1, dn, w, arch)
        self.discriminator = LayerMaker().make(self.arch.discriminator)
        # define customized optimizers
        self.generator_optimizer = None
        self.generator_lr_scheduler = None
        ###
        self.discriminator_optimizer = None
        self.discriminator_lr_scheduler = None

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

    def set_training_params(self, training_params=None):
        if training_params is None:
            self.training_params = {
                key: {"lr": 0.01, "factor": 0.9, "patience": 10000, "min_lr": 0.00001}
                for key in ["reconstructor", "generator", "discriminator"]
            }
        else:
            self.training_params = training_params

    def initialize_training_components(self):
        super(AdversarialAutoencoder, self).initialize_training_components()
        # define customized optimizers
        self.generator_optimizer = optim.SGD(
            [
                {"params": self.vectorizer.parameters()},
                {"params": self.encoder.parameters()},
                {"params": self.discriminator.parameters()},
            ],
            lr=self.training_params["generator"]["lr"],
        )
        self.generator_lr_scheduler = CustomLRScheduler(
            self.generator_optimizer,
            factor=self.training_params["generator"]["factor"],
            patience=self.training_params["generator"]["patience"],
            min_lr=self.training_params["generator"]["min_lr"],
        )
        ###
        self.discriminator_optimizer = optim.SGD(
            [{"params": self.discriminator.parameters()}],
            lr=self.training_params["discriminator"]["lr"],
        )
        self.discriminator_lr_scheduler = CustomLRScheduler(
            self.discriminator_optimizer,
            factor=self.training_params["discriminator"]["factor"],
            patience=self.training_params["discriminator"]["patience"],
            min_lr=self.training_params["discriminator"]["min_lr"],
        )

    def transform_input(self, input_vals, device, ks=2, input_noise=0.0):
        # scans by sliding window of w
        assert isinstance(input_vals, Tensor)
        input_vals = unfold(
            tensor(input_vals, device=device).T[None, None, :, :], kernel_size=(ks, self.w)
        )[0].T
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

    def train_batch(self, input_vals, device, input_noise=0.0):
        """
        Training for one batch of data, this will move into autoencoder module
        :param input_vals:
        :param device:
        :param input_noise:
        :return:
        """
        self.train()
        self.input_ndx, self.target_vals, self.one_hot_input = self.transform_input(
            input_vals, device, input_noise=input_noise
        )
        # train encoder_decoder
        self.reconstructor_optimizer.zero_grad()
        reconstructor_output = self.forward_encoder_decoder(self.one_hot_input)
        reconstructor_loss = self.criterion_NLLLoss(
            reconstructor_output, self.input_ndx.reshape((-1,))
        )
        reconstructor_loss.backward()
        self.reconstructor_optimizer.step()
        wandb.log({"reconstructor_loss": reconstructor_loss.item()})
        wandb.log({"reconstructor_LR": self.reconstructor_lr_scheduler.get_last_lr()})
        self.training_params["reconstructor"]["lr"] = self.reconstructor_lr_scheduler.get_last_lr()
        self.reconstructor_lr_scheduler.step(reconstructor_loss.item())
        # train generator
        self.generator_optimizer.zero_grad()
        generator_output = self.forward_generator(self.one_hot_input)
        generator_loss = self.criterion_NLLLoss(
            generator_output,
            zeros((generator_output.shape[0],), device=device).long(),
        )
        generator_loss.backward()
        self.generator_optimizer.step()
        wandb.log({"generator_loss": generator_loss.item()})
        wandb.log({"generator_LR": self.generator_lr_scheduler.get_last_lr()})
        self.training_params["generator"]["lr"] = self.generator_lr_scheduler.get_last_lr()
        # train discriminator
        self.discriminator_optimizer.zero_grad()
        ndx = randperm(self.w)
        discriminator_output = self.forward_discriminator(self.one_hot_input[:, ndx, :])
        discriminator_loss = self.criterion_NLLLoss(
            discriminator_output,
            ones((discriminator_output.shape[0],), device=device).long(),
        )
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        wandb.log({"discriminator_loss": discriminator_loss.item()})
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

    def test_batch(self, input_vals, device, input_noise=0.0):
        """
        Test a single batch of data, this will move into autoencoder
        :param input_vals:
        :return:
        """
        self.eval()
        with no_grad():
            input_ndx, _, one_hot_input = self.transform_input(
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
            wandb.log({"test_reconstructor_loss": reconstructor_loss.item()})
            wandb.log({"test_generator_loss": generator_loss.item()})
            wandb.log({"test_reconstructor_accuracy": reconstructor_accuracy.item()})
            wandb.log({"test_consensus_accuracy": consensus_seq_acc})
            # clean up
            del reconstructor_output
            del generator_output
            del reconstructor_loss
            del generator_loss
