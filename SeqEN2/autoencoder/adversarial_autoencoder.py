#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


import wandb
from torch import argmax, float32
from torch import load as torch_load
from torch import no_grad, ones, optim, randperm
from torch import save as torch_save
from torch import sum as torch_sum
from torch import tensor, transpose, zeros
from torch.nn import Module, MSELoss, NLLLoss
from torch.nn.functional import one_hot

from SeqEN2.autoencoder.utils import Architecture, CustomLRScheduler, LayerMaker


# class for AAE
class AdversarialAutoencoder(Module):
    def __init__(self, d0, d1, dn, w, arch):
        super(AdversarialAutoencoder, self).__init__()
        self.d0 = d0
        self.d1 = d1
        self.dn = dn
        self.w = w
        self.arch = Architecture(arch)
        self.vectorizer = LayerMaker().make(self.arch.vectorizer)
        self.encoder = LayerMaker().make(self.arch.encoder)
        self.decoder = LayerMaker().make(self.arch.decoder)
        self.devectorizer = LayerMaker().make(self.arch.devectorizer)
        self.classifier = LayerMaker().make(self.arch.classifier)
        self.discriminator = LayerMaker().make(self.arch.discriminator)
        # training components
        self.training_params = None
        # define customized optimizers
        self.reconstructor_optimizer = None
        self.reconstructor_lr_scheduler = None
        ###
        self.generator_optimizer = None
        self.generator_lr_scheduler = None
        ###
        self.discriminator_optimizer = None
        self.discriminator_lr_scheduler = None
        ###
        self.classifier_optimizer = None
        self.classifier_lr_scheduler = None
        # Loss functions
        self.criterion_NLLLoss = NLLLoss()
        self.criterion_MSELoss = MSELoss()

    def forward_encoder_decoder(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        decoded = transpose(self.decoder(encoded), 1, 2).reshape(-1, self.d1)
        devectorized = self.devectorizer(decoded)
        return devectorized

    def forward_generator(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        discriminator_output = self.discriminator(encoded)
        return discriminator_output

    def forward_discriminator(self, one_hot_input):
        return self.forward_generator(one_hot_input)

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

    def save(self, model_dir, epoch):
        torch_save(self.vectorizer, model_dir / f"vectorizer_{epoch}.m")
        torch_save(self.encoder, model_dir / f"encoder_{epoch}.m")
        torch_save(self.decoder, model_dir / f"decoder_{epoch}.m")
        torch_save(self.devectorizer, model_dir / f"devectorizer_{epoch}.m")
        torch_save(self.classifier, model_dir / f"classifier_{epoch}.m")
        torch_save(self.discriminator, model_dir / f"discriminator_{epoch}.m")

    def load(self, model_dir, version, map_location):
        self.vectorizer = torch_load(
            model_dir / f"vectorizer_{version}.m", map_location=map_location
        )
        self.encoder = torch_load(
            model_dir / f"encoder_{version}.m", map_location=map_location
        )
        self.decoder = torch_load(
            model_dir / f"decoder_{version}.m", map_location=map_location
        )
        self.devectorizer = torch_load(
            model_dir / f"devectorizer_{version}.m", map_location=map_location
        )
        self.classifier = torch_load(
            model_dir / f"classifier_{version}.m", map_location=map_location
        )
        self.discriminator = torch_load(
            model_dir / f"discriminator_{version}.m", map_location=map_location
        )

    def initialize_training_components(self, training_params=None):
        if training_params is None:
            self.training_params = {
                key: {"lr": 0.01, "factor": 0.99, "patience": 10000, "min_lr": 0.00001}
                for key in ["reconstructor", "generator", "discriminator", "classifier"]
            }
        else:
            self.training_params = training_params
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
        ###
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
        ###
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

    def train_batch(self, input_vals, device):
        '''
        Training for one batch of data, this will move into autoencoder module
        :param input_vals:
        :return:
        '''
        self.train()
        input_ndx = tensor(input_vals[:, : self.w], device=device).long()
        one_hot_input = one_hot(input_ndx, num_classes=self.d0) * 1.0
        # train encoder_decoder
        self.reconstructor_optimizer.zero_grad()
        reconstructor_output = self.forward_encoder_decoder(one_hot_input)
        reconstructor_loss = self.criterion_NLLLoss(
            reconstructor_output, input_ndx.reshape((-1,))
        )
        reconstructor_loss.backward()
        self.reconstructor_optimizer.step()
        wandb.log({"reconstructor_loss": reconstructor_loss.item()})
        wandb.log(
            {
                "reconstructor_LR": self.reconstructor_lr_scheduler.get_last_lr()
            }
        )
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
        wandb.log({"generator_loss": generator_loss.item()})
        wandb.log(
            {"generator_LR": self.generator_lr_scheduler.get_last_lr()}
        )
        # train discriminator
        self.discriminator_optimizer.zero_grad()
        ndx = randperm(self.w)
        discriminator_output = self.forward_discriminator(
            one_hot_input[:, ndx, :]
        )
        discriminator_loss = self.criterion_NLLLoss(
            discriminator_output,
            ones((discriminator_output.shape[0],), device=device).long(),
        )
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        wandb.log({"discriminator_loss": discriminator_loss.item()})
        wandb.log(
            {
                "discriminator_LR": self.discriminator_lr_scheduler.get_last_lr()
            }
        )
        gen_disc_loss = 0.5 * (generator_loss.item() + discriminator_loss.item())
        self.generator_lr_scheduler.step(gen_disc_loss)
        self.discriminator_lr_scheduler.step(gen_disc_loss)
        # train classifier
        self.classifier_optimizer.zero_grad()
        classifier_target = tensor(
            input_vals[:, self.w :], device=device, dtype=float32
        )
        classifier_output = self.forward_classifier(one_hot_input)
        classifier_loss = self.criterion_MSELoss(
            classifier_output, classifier_target
        )
        classifier_loss.backward()
        self.classifier_optimizer.step()
        wandb.log({"classifier_loss": classifier_loss.item()})
        wandb.log(
            {"classifier_LR": self.classifier_lr_scheduler.get_last_lr()}
        )
        self.classifier_lr_scheduler.step(classifier_loss.item())

    def test_batch(self, input_vals, device):
        '''
        Test a single batch of data, this will move into autoencoder
        :param input_vals:
        :return:
        '''
        with no_grad():
            input_ndx = tensor(input_vals[:, : self.w], device=device).long()
            one_hot_input = one_hot(input_ndx, num_classes=self.d0) * 1.0
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
            classifier_target = tensor(
                input_vals[:, self.w :], device=device, dtype=float32
            )
            classifier_loss = self.criterion_MSELoss(
                classifier_output, classifier_target
            )
            # reconstructor acc
            reconstructor_ndx = argmax(reconstructor_output, dim=1)
            reconstructor_accuracy = (
                torch_sum(reconstructor_ndx == input_ndx.reshape((-1,)))
                / reconstructor_ndx.shape[0]
            )
            # reconstruction_loss, discriminator_loss, classifier_loss
            wandb.log({"test_reconstructor_loss": reconstructor_loss.item()})
            wandb.log({"test_generator_loss": generator_loss.item()})
            wandb.log({"test_classifier_loss": classifier_loss.item()})
            wandb.log({"test_reconstructor_accuracy": reconstructor_accuracy.item()})