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

    def forward_eval_embed(self, one_hot_input):
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
        assert isinstance(input_vals, Tensor), f"expected Tensor type, received {type(input_vals)}"
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
        assert isinstance(input_vals, Tensor), f"expected Tensor type, received {type(input_vals)}"
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
        assert isinstance(input_vals, Tensor), f"expected Tensor type, received {type(input_vals)}"
        kernel_size = (input_vals.shape[1], self.w)
        input_vals = unfold(input_vals.float().T[None, None, :, :], kernel_size=kernel_size)[0].T
        input_ndx = input_vals[:, : self.w].long()
        target_vals_ss = input_vals[:, self.w : -self.w].long()
        target_vals_cl = input_vals[:, -self.w :].mean(axis=1).reshape((-1, 1))
        target_vals_cl = cat((target_vals_cl, 1 - target_vals_cl), 1).float()
        one_hot_input = one_hot(input_ndx, num_classes=self.d0) * 1.0
        if input_noise > 0.0:
            ndx = randperm(self.w)
            size = list(one_hot_input.shape)
            size[-1] = 1
            p = tensor(choice([1, 0], p=[input_noise, 1 - input_noise], size=size)).to(device)
            mutated_one_hot = (one_hot_input[:, ndx, :] * p) + (one_hot_input * (1 - p))
            return input_ndx, target_vals_ss, target_vals_cl, mutated_one_hot
        else:
            return input_ndx, target_vals_ss, target_vals_cl, one_hot_input

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
        # train encoder_decoder
        self.reconstructor_optimizer.zero_grad()
        reconstructor_output = self.forward_encoder_decoder(one_hot_input)
        reconstructor_loss = self.criterion_NLLLoss(reconstructor_output, input_ndx.reshape((-1,)))
        reconstructor_loss.backward()
        self.reconstructor_optimizer.step()
        self._training_settings.reconstructor.lr = self.reconstructor_lr_scheduler.get_last_lr()
        self.reconstructor_lr_scheduler.step(reconstructor_loss.item())
        self.log("reconstructor_loss", reconstructor_loss.item())
        self.log("reconstructor_LR", self.training_settings.reconstructor.lr)
        # train for continuity
        if not self.ignore_continuity:
            self.continuity_optimizer.zero_grad()
            encoded_output = self.forward_embed(one_hot_input)
            continuity_loss_r = self.criterion_MSELoss(
                encoded_output, cat((encoded_output[1:], encoded_output[-1].unsqueeze(0)), 0)
            )
            continuity_loss_l = self.criterion_MSELoss(
                encoded_output, cat((encoded_output[0].unsqueeze(0), encoded_output[:-1]), 0)
            )
            continuity_loss = continuity_loss_r + continuity_loss_l
            continuity_loss.backward()
            self.continuity_optimizer.step()
            self._training_settings.continuity.lr = self.continuity_lr_scheduler.get_last_lr()
            self.continuity_lr_scheduler.step(continuity_loss.item())
            self.log("continuity_loss", continuity_loss.item())
            self.log("continuity_LR", self.training_settings.continuity.lr)
            del encoded_output
            del continuity_loss
        # train generator
        self.generator_optimizer.zero_grad()
        generator_output = self.forward_generator(one_hot_input)
        generator_loss = self.criterion_NLLLoss(
            generator_output,
            zeros((generator_output.shape[0],), device=device).long(),
        )
        generator_loss.backward()
        self.generator_optimizer.step()
        self._training_settings.generator.lr = self.generator_lr_scheduler.get_last_lr()
        self.log("generator_loss", generator_loss.item())
        self.log("generator_LR", self.training_settings.generator.lr)
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
        self._training_settings.discriminator.lr = self.discriminator_lr_scheduler.get_last_lr()
        gen_disc_loss = 0.5 * (generator_loss.item() + discriminator_loss.item())
        self.generator_lr_scheduler.step(gen_disc_loss)
        self.discriminator_lr_scheduler.step(gen_disc_loss)
        self.log("discriminator_loss", discriminator_loss.item())
        self.log("discriminator_LR", self.training_settings.discriminator.lr)
        # train classifier
        self.classifier_optimizer.zero_grad()
        classifier_output = self.forward_classifier(one_hot_input)
        classifier_loss = self.criterion_MSELoss(classifier_output, target_vals)
        classifier_loss.backward()
        self.classifier_optimizer.step()
        self._training_settings.classifier.lr = self.classifier_lr_scheduler.get_last_lr()
        self.classifier_lr_scheduler.step(classifier_loss.item())
        self.log("classifier_loss", classifier_loss.item())
        self.log("classifier_LR", self._training_settings.classifier.lr)
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
        self._training_settings.reconstructor.lr = self.reconstructor_lr_scheduler.get_last_lr()
        self.reconstructor_lr_scheduler.step(reconstructor_loss.item())
        self.log("reconstructor_loss", reconstructor_loss.item())
        self.log("reconstructor_LR", self.training_settings.reconstructor.lr)
        # train for continuity
        if not self.ignore_continuity:
            self.continuity_optimizer.zero_grad()
            encoded_output = self.forward_embed(one_hot_input)
            continuity_loss_r = self.criterion_MSELoss(
                encoded_output, cat((encoded_output[1:], encoded_output[-1].unsqueeze(0)), 0)
            )
            continuity_loss_l = self.criterion_MSELoss(
                encoded_output, cat((encoded_output[0].unsqueeze(0), encoded_output[:-1]), 0)
            )
            continuity_loss = continuity_loss_r + continuity_loss_l
            continuity_loss.backward()
            self.continuity_optimizer.step()
            self._training_settings.continuity.lr = self.continuity_lr_scheduler.get_last_lr()
            self.continuity_lr_scheduler.step(continuity_loss.item())
            self.log("continuity_loss", continuity_loss.item())
            self.log("continuity_LR", self.training_settings.continuity.lr)
            del encoded_output
            del continuity_loss
        # train encoder_SS_decoder
        self.ss_decoder_optimizer.zero_grad()
        ss_decoder_output = self.forward_ss_decoder(one_hot_input)
        ss_decoder_loss = self.criterion_NLLLoss(ss_decoder_output, target_vals.reshape((-1,)))
        ss_decoder_loss.backward()
        self.ss_decoder_optimizer.step()
        self._training_settings.ss_decoder.lr = self.ss_decoder_lr_scheduler.get_last_lr()
        self.ss_decoder_lr_scheduler.step(ss_decoder_loss.item())
        self.log("ss_decoder_loss", ss_decoder_loss.item())
        self.log("ss_decoder_LR", self.training_settings.ss_decoder.lr)
        # train generator
        self.generator_optimizer.zero_grad()
        generator_output = self.forward_generator(one_hot_input)
        generator_loss = self.criterion_NLLLoss(
            generator_output,
            zeros((generator_output.shape[0],), device=device).long(),
        )
        generator_loss.backward()
        self.generator_optimizer.step()
        self._training_settings.generator.lr = self.generator_lr_scheduler.get_last_lr()
        self.log("generator_loss", generator_loss.item())
        self.log("generator_LR", self.training_settings.generator.lr)
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
        self._training_settings.discriminator.lr = self.discriminator_lr_scheduler.get_last_lr()
        gen_disc_loss = 0.5 * (generator_loss.item() + discriminator_loss.item())
        self.generator_lr_scheduler.step(gen_disc_loss)
        self.discriminator_lr_scheduler.step(gen_disc_loss)
        self.log("discriminator_loss", discriminator_loss.item())
        self.log("discriminator_LR", self.training_settings.discriminator.lr)
        # clean up
        del reconstructor_loss
        del reconstructor_output
        del generator_output
        del generator_loss
        del discriminator_output
        del discriminator_loss
        del ss_decoder_output
        del ss_decoder_loss

        # training with clss data
        if "clss" in input_vals.keys():
            input_ndx, target_vals_ss, target_vals_cl, one_hot_input = self.transform_input_clss(
                input_vals["clss"], device, input_noise=input_noise
            )
            # train encoder_decoder
            self.reconstructor_optimizer.zero_grad()
            reconstructor_output = self.forward_encoder_decoder(one_hot_input)
            reconstructor_loss = self.criterion_NLLLoss(
                reconstructor_output, input_ndx.reshape((-1,))
            )
            reconstructor_loss.backward()
            self.reconstructor_optimizer.step()
            self._training_settings.reconstructor.lr = self.reconstructor_lr_scheduler.get_last_lr()
            self.reconstructor_lr_scheduler.step(reconstructor_loss.item())
            self.log("reconstructor_loss", reconstructor_loss.item())
            self.log("reconstructor_LR", self.training_settings.reconstructor.lr)
            # train for continuity
            if not self.ignore_continuity:
                self.continuity_optimizer.zero_grad()
                encoded_output = self.forward_embed(one_hot_input)
                continuity_loss_r = self.criterion_MSELoss(
                    encoded_output, cat((encoded_output[1:], encoded_output[-1].unsqueeze(0)), 0)
                )
                continuity_loss_l = self.criterion_MSELoss(
                    encoded_output, cat((encoded_output[0].unsqueeze(0), encoded_output[:-1]), 0)
                )
                continuity_loss = continuity_loss_r + continuity_loss_l
                continuity_loss.backward()
                self.continuity_optimizer.step()
                self._training_settings.continuity.lr = self.continuity_lr_scheduler.get_last_lr()
                self.continuity_lr_scheduler.step(continuity_loss.item())
                self.log("continuity_loss", continuity_loss.item())
                self.log("continuity_LR", self.training_settings.continuity.lr)
                del encoded_output
                del continuity_loss
            # train encoder_SS_decoder
            self.ss_decoder_optimizer.zero_grad()
            ss_decoder_output = self.forward_ss_decoder(one_hot_input)
            ss_decoder_loss = self.criterion_NLLLoss(
                ss_decoder_output, target_vals_ss.reshape((-1,))
            )
            ss_decoder_loss.backward()
            self.ss_decoder_optimizer.step()
            self._training_settings.ss_decoder.lr = self.ss_decoder_lr_scheduler.get_last_lr()
            self.ss_decoder_lr_scheduler.step(ss_decoder_loss.item())
            self.log("ss_decoder_loss", ss_decoder_loss.item())
            self.log("ss_decoder_LR", self.training_settings.ss_decoder.lr)
            # train generator
            self.generator_optimizer.zero_grad()
            generator_output = self.forward_generator(one_hot_input)
            generator_loss = self.criterion_NLLLoss(
                generator_output,
                zeros((generator_output.shape[0],), device=device).long(),
            )
            generator_loss.backward()
            self.generator_optimizer.step()
            self._training_settings.generator.lr = self.generator_lr_scheduler.get_last_lr()
            self.log("generator_loss", generator_loss.item())
            self.log("generator_LR", self.training_settings.generator.lr)
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
            self._training_settings.discriminator.lr = self.discriminator_lr_scheduler.get_last_lr()
            gen_disc_loss = 0.5 * (generator_loss.item() + discriminator_loss.item())
            self.generator_lr_scheduler.step(gen_disc_loss)
            self.discriminator_lr_scheduler.step(gen_disc_loss)
            self.log("discriminator_loss", discriminator_loss.item())
            self.log("discriminator_LR", self.training_settings.discriminator.lr)
            # train classifier
            self.classifier_optimizer.zero_grad()
            classifier_output = self.forward_classifier(one_hot_input)
            classifier_loss = self.criterion_MSELoss(classifier_output, target_vals_cl)
            classifier_loss.backward()
            self.classifier_optimizer.step()
            self._training_settings.classifier.lr = self.classifier_lr_scheduler.get_last_lr()
            self.classifier_lr_scheduler.step(classifier_loss.item())
            self.log("classifier_loss", classifier_loss.item())
            self.log("classifier_LR", self._training_settings.classifier.lr)
            # clean up
            del reconstructor_loss
            del reconstructor_output
            del generator_output
            del generator_loss
            del discriminator_output
            del discriminator_loss
            del classifier_output
            del classifier_loss
            del ss_decoder_output
            del ss_decoder_loss

    def test_batch(self, input_vals, device):
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
            (
                reconstructor_output,
                generator_output,
                classifier_output,
                ss_decoder_output,
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
            # test for continuity
            if not self.ignore_continuity:
                encoded_output = self.forward_embed(one_hot_input)
                continuity_loss_r = self.criterion_MSELoss(
                    encoded_output, cat((encoded_output[1:], encoded_output[-1].unsqueeze(0)), 0)
                )
                continuity_loss_l = self.criterion_MSELoss(
                    encoded_output, cat((encoded_output[0].unsqueeze(0), encoded_output[:-1]), 0)
                )
                continuity_loss = continuity_loss_r + continuity_loss_l
                self.log("test_continuity_loss", continuity_loss.item())
            # testing with ss data
            input_ndx, target_vals, one_hot_input = self.transform_input_ss(
                input_vals["ss"], device
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
            ss_decoder_loss = self.criterion_NLLLoss(ss_decoder_output, target_vals.reshape((-1,)))
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
            self.log("test_reconstructor_loss", reconstructor_loss.item())
            self.log("test_ss_decoder_loss", ss_decoder_loss.item())
            self.log("test_reconstructor_accuracy", reconstructor_accuracy.item())
            self.log("test_consensus_accuracy", consensus_seq_acc)
            self.log("test_ss_decoder_accuracy", ss_decoder_accuracy.item())
            self.log("test_consensus_ss_accuracy", consensus_ss_acc)
            # test for continuity
            if not self.ignore_continuity:
                encoded_output = self.forward_embed(one_hot_input)
                continuity_loss_r = self.criterion_MSELoss(
                    encoded_output, cat((encoded_output[1:], encoded_output[-1].unsqueeze(0)), 0)
                )
                continuity_loss_l = self.criterion_MSELoss(
                    encoded_output, cat((encoded_output[0].unsqueeze(0), encoded_output[:-1]), 0)
                )
                continuity_loss = continuity_loss_r + continuity_loss_l
                self.log("test_continuity_loss", continuity_loss.item())
            # training with clss data
            if "clss" in input_vals.keys():
                (
                    input_ndx,
                    target_vals_ss,
                    target_vals_cl,
                    one_hot_input,
                ) = self.transform_input_clss(input_vals["clss"], device)
                (
                    reconstructor_output,
                    generator_output,
                    classifier_output,
                    ss_decoder_output,
                ) = self.forward_test(one_hot_input)
                reconstructor_loss = self.criterion_NLLLoss(
                    reconstructor_output, input_ndx.reshape((-1,))
                )
                classifier_loss = self.criterion_MSELoss(classifier_output, target_vals_cl)
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
                    torch_sum(ss_decoder_ndx == target_vals_ss.reshape((-1,)))
                    / ss_decoder_ndx.shape[0]
                )
                consensus_ss_acc, _ = consensus_acc(
                    target_vals_ss, ss_decoder_ndx.reshape((-1, self.w)), device
                )
                # reconstruction_loss, discriminator_loss, classifier_loss
                self.log("test_reconstructor_loss", reconstructor_loss.item())
                self.log("test_classifier_loss", classifier_loss.item())
                self.log("test_reconstructor_accuracy", reconstructor_accuracy.item())
                self.log("test_consensus_accuracy", consensus_seq_acc)
                self.log("test_ss_decoder_loss", ss_decoder_loss.item())
                self.log("test_ss_decoder_accuracy", ss_decoder_accuracy.item())
                self.log("test_consensus_ss_accuracy", consensus_ss_acc)
                # test for continuity
                if not self.ignore_continuity:
                    encoded_output = self.forward_embed(one_hot_input)
                    continuity_loss_r = self.criterion_MSELoss(
                        encoded_output,
                        cat((encoded_output[1:], encoded_output[-1].unsqueeze(0)), 0),
                    )
                    continuity_loss_l = self.criterion_MSELoss(
                        encoded_output,
                        cat((encoded_output[0].unsqueeze(0), encoded_output[:-1]), 0),
                    )
                    continuity_loss = continuity_loss_r + continuity_loss_l
                    self.log("test_continuity_loss", continuity_loss.item())

    def embed_batch(self, input_vals, device, dataset="cl"):
        """
        Test a single batch of data, this will move into autoencoder
        :param dataset:
        :param device:
        :param input_vals:
        :return:
        """
        assert isinstance(input_vals, Tensor), "AAECSS requires a tensor as input_vals"
        self.eval()
        with no_grad():
            if dataset == "cl":
                # testing with cl data
                input_ndx, target_vals, one_hot_input = self.transform_input_cl(input_vals, device)
                (
                    embedding,
                    classifier_output,
                    ss_decoder_output,
                ) = self.forward_eval_embed(one_hot_input)
                consensus_ss = get_consensus_seq(
                    argmax(ss_decoder_output, dim=1).reshape((-1, self.w)), device
                )
                return embedding, classifier_output, consensus_ss
            elif dataset == "clss":
                # testing with clss data
                (
                    input_ndx,
                    target_vals_ss,
                    target_vals_cl,
                    one_hot_input,
                ) = self.transform_input_clss(input_vals, device)
                (
                    embedding,
                    classifier_output,
                    ss_decoder_output,
                ) = self.forward_eval_embed(one_hot_input)
                consensus_ss = get_consensus_seq(
                    argmax(ss_decoder_output, dim=1).reshape((-1, self.w)), device
                )
                return embedding, classifier_output, consensus_ss
