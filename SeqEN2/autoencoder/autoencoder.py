#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


from numpy.random import choice
from torch import argmax, cat
from torch import load as torch_load
from torch import no_grad, optim, randperm
from torch import save as torch_save
from torch import sum as torch_sum
from torch import tensor, transpose
from torch.nn import Module, NLLLoss
from torch.nn.functional import one_hot, unfold

import wandb
from SeqEN2.autoencoder.utils import Architecture, CustomLRScheduler, LayerMaker
from SeqEN2.utils.seq_tools import consensus_acc
from SeqEN2.utils.utils import get_map_location


# class for AE
class Autoencoder(Module):

    aa_keys = "WYFMILVAGPSTCEDQNHRK*"  # amino acids class labels
    d0 = 21  # amino acids class size

    def __init__(self, d1, dn, w, arch):
        super(Autoencoder, self).__init__()
        self.d1 = d1
        self.dn = dn
        self.w = w
        self.arch = Architecture(arch)
        self.vectorizer = LayerMaker().make(self.arch.vectorizer)
        self.encoder = LayerMaker().make(self.arch.encoder)
        self.decoder = LayerMaker().make(self.arch.decoder)
        self.devectorizer = LayerMaker().make(self.arch.devectorizer)
        # training components
        self.training_params = None
        # define customized optimizers
        self.reconstructor_optimizer = None
        self.reconstructor_lr_scheduler = None
        # Loss functions
        self.criterion_NLLLoss = NLLLoss()
        # training inputs placeholders
        self.input_ndx = None
        self.target_vals = None
        self.one_hot_input = None

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

    def load(self, model_dir, version):
        self.vectorizer = torch_load(
            model_dir / f"vectorizer_{version}.m", map_location=get_map_location()
        )
        self.encoder = torch_load(
            model_dir / f"encoder_{version}.m", map_location=get_map_location()
        )
        self.decoder = torch_load(
            model_dir / f"decoder_{version}.m", map_location=get_map_location()
        )
        self.devectorizer = torch_load(
            model_dir / f"devectorizer_{version}.m", map_location=get_map_location()
        )

    def transform_input(self, input_vals, device, input_noise=0.0):
        # scans by sliding window of w
        input_vals = unfold(
            tensor(input_vals, device=device).T[None, None, :, :], kernel_size=(2, self.w)
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

    def set_training_params(self, training_params=None):
        if training_params is None:
            self.training_params = {
                key: {"lr": 0.01, "factor": 0.9, "patience": 10000, "min_lr": 0.00001}
                for key in ["reconstructor"]
            }
        else:
            self.training_params = training_params

    def initialize_training_components(self):
        # define customized optimizers
        self.reconstructor_optimizer = optim.SGD(
            [
                {"params": self.vectorizer.parameters()},
                {"params": self.encoder.parameters()},
                {"params": self.decoder.parameters()},
                {"params": self.devectorizer.parameters()},
            ],
            lr=self.training_params["reconstructor"]["lr"],
        )
        self.reconstructor_lr_scheduler = CustomLRScheduler(
            self.reconstructor_optimizer,
            factor=self.training_params["reconstructor"]["factor"],
            patience=self.training_params["reconstructor"]["patience"],
            min_lr=self.training_params["reconstructor"]["min_lr"],
        )

    def initialize_for_training(self, training_params):
        self.set_training_params(training_params=training_params)
        self.initialize_training_components()

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
        # clean up
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
            input_ndx, _, one_hot_input = self.transform_input(input_vals, device, input_noise=0.0)
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
            wandb.log({"test_reconstructor_loss": reconstructor_loss.item()})
            wandb.log({"test_reconstructor_accuracy": reconstructor_accuracy.item()})
            wandb.log({"test_consensus_accuracy": consensus_seq_acc})
            # clean up
            del reconstructor_loss
            del reconstructor_output
