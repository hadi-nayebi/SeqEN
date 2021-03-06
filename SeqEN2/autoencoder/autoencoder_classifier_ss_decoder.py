#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from typing import Dict

from torch import no_grad, transpose

from SeqEN2.autoencoder.autoencoder_classifier import AutoencoderClassifier
from SeqEN2.autoencoder.autoencoder_ss_decoder import AutoencoderSSDecoder
from SeqEN2.autoencoder.utils import print_shapes
from SeqEN2.utils.custom_dataclasses import AECSSTrainingSettings
from SeqEN2.utils.seq_tools import output_to_ndx


# class for AAE
class AutoencoderClassifierSSDecoder(AutoencoderClassifier, AutoencoderSSDecoder):
    def __init__(self, d1, dn, w, arch):
        super(AutoencoderClassifierSSDecoder, self).__init__(d1, dn, w, arch)
        # training components
        self._training_settings = AECSSTrainingSettings()

    @property
    def training_settings(self) -> AECSSTrainingSettings:
        return self._training_settings

    @training_settings.setter
    def training_settings(self, value=None) -> None:
        if isinstance(value, Dict) or value is None or isinstance(value, AECSSTrainingSettings):
            if isinstance(value, Dict):
                try:
                    self._training_settings = AECSSTrainingSettings(**value)
                except TypeError as e:
                    raise KeyError(f"missing/extra keys for AECSSTrainingSettings, {e}")
            elif isinstance(value, AECSSTrainingSettings):
                self._training_settings = value
        else:
            raise TypeError(
                f"Training settings must be a dict or None or type AECSSTrainingSettings, {type(value)} is passed."
            )

    def forward_test(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        decoded = transpose(self.decoder(encoded), 1, 2).reshape(-1, self.d1)
        devectorized = self.devectorizer(decoded)
        classifier_output = self.classifier(encoded)
        ss_decoder_output = transpose(self.ss_decoder(encoded), 1, 2).reshape(-1, self.ds)
        return devectorized, classifier_output, ss_decoder_output, encoded

    def unit_test_forward(self, one_hot_input):
        vectorized = print_shapes(
            one_hot_input.reshape((-1, self.d0)), self.vectorizer, "vectorizer"
        )
        encoded = print_shapes(
            transpose(vectorized.reshape((-1, self.w, self.d1)), 1, 2), self.encoder, "encoder"
        )
        decoded = transpose(print_shapes(encoded, self.decoder, "decoder"), 1, 2).reshape(
            -1, self.d1
        )
        devectorized = print_shapes(decoded, self.devectorizer, "devectorizer")
        classifier_output = print_shapes(encoded, self.classifier, "classifier")
        ss_decoder_output = transpose(
            print_shapes(encoded, self.ss_decoder, "ss_decoder"), 1, 2
        ).reshape(-1, self.ds)
        return devectorized, classifier_output, ss_decoder_output, encoded

    def train_focused(self, **kwargs):
        self.focused_optimizer.zero_grad()
        loss = None
        if self.focus in ["vectorizer", "encoder", "decoder", "devectorizer"]:
            loss = self.autoencoder_focused(**kwargs)
        elif self.focus == "classifier":
            loss = self.classifier_focused(**kwargs)
        elif self.focus == "ss_decoder":
            loss = self.ss_decoder_focused(**kwargs)
        if loss is not None:
            self.focused_optimizer.step()
            self._modular_training_settings.focused.lr = self.focused_lr_scheduler.get_last_lr()
            self.focused_lr_scheduler.step(loss.item())
            self.log(f"focused_{self.focus}_LR", self.focused_lr_scheduler.get_last_lr())

    def train_one_batch(self, input_vals, input_noise=0.0, device=None, input_keys="ASC"):
        if input_vals is not None:
            input_ndx, target_vals_ss, target_vals_cl, one_hot_input = self.transform_input(
                input_vals, device, input_noise=input_noise, input_keys=input_keys
            )
            self.train_reconstructor(one_hot_input, input_ndx)
            if "C" in input_keys:
                self.train_classifier(one_hot_input, target_vals_cl)
            if "S" in input_keys:
                self.train_ss_decoder(one_hot_input, target_vals_ss)
            if self.focus is not None:
                self.train_focused(
                    one_hot_input=one_hot_input,
                    input_ndx=input_ndx,
                    target_vals_cl=target_vals_cl,
                    target_vals_ss=target_vals_ss,
                    input_keys=input_keys,
                )

    @staticmethod
    def assert_input_type(input_vals):
        assert isinstance(input_vals, Dict), "AECSS requires a dict as input_vals"

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
                input_vals["clss"], input_noise=input_noise, device=device, input_keys="ASC"
            )

    def test_one_batch(self, input_vals, device, input_keys="ASC"):
        if input_vals is not None:
            input_ndx, target_vals_ss, target_vals_cl, one_hot_input = self.transform_input(
                input_vals, device, input_keys=input_keys
            )
            (
                reconstructor_output,
                classifier_output,
                ss_decoder_output,
                encoded_output,
            ) = self.forward_test(one_hot_input)
            self.test_reconstructor(reconstructor_output, input_ndx, device)
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
                self.test_one_batch(input_vals["clss"], device, input_keys="ASC")

    def eval_one_batch(self, input_vals, device, input_keys="A--", embed_only=False):
        if input_vals is not None:
            _, _, _, one_hot_input = self.transform_input(input_vals, device, input_keys=input_keys)
            if embed_only:
                encoded_output = self.forward_embed(one_hot_input)
                return {"embedding": encoded_output}
            else:
                (
                    reconstructor_output,
                    classifier_output,
                    ss_decoder_output,
                    encoded_output,
                ) = self.forward_test(one_hot_input)
                return {
                    "reconstructor_output": output_to_ndx(reconstructor_output, self.w),
                    "classifier_output": classifier_output,
                    "ss_decoder_output": output_to_ndx(ss_decoder_output, self.w),
                    "embedding": encoded_output,
                }
