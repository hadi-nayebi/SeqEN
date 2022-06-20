#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from typing import Dict

from torch import load as torch_load
from torch import no_grad, optim
from torch import save as torch_save
from torch import transpose
from torch.nn.utils import clip_grad_value_

from SeqEN2.autoencoder.autoencoder import Autoencoder
from SeqEN2.autoencoder.utils import CustomLRScheduler, LayerMaker
from SeqEN2.utils.custom_dataclasses import AESSTrainingSettings
from SeqEN2.utils.seq_tools import consensus_acc, output_to_ndx, reconstructor_acc
from SeqEN2.utils.utils import get_map_location


# class for AAE
class AutoencoderSSDecoder(Autoencoder):
    ds = 9  # SS labels dimension

    def __init__(self, d1, dn, w, arch):
        super(AutoencoderSSDecoder, self).__init__(d1, dn, w, arch)
        assert self.arch.ss_decoder is not None, "arch missing ss_decoder."
        self.ss_decoder = LayerMaker().make(self.arch.ss_decoder)
        # training components
        self._training_settings = AESSTrainingSettings()
        ###
        self.ss_decoder_optimizer = None
        self.ss_decoder_lr_scheduler = None

    @property
    def training_settings(self) -> AESSTrainingSettings:
        return self._training_settings

    @training_settings.setter
    def training_settings(self, value=None) -> None:
        if isinstance(value, Dict) or value is None or isinstance(value, AESSTrainingSettings):
            if isinstance(value, Dict):
                try:
                    self._training_settings = AESSTrainingSettings(**value)
                except TypeError as e:
                    raise KeyError(f"missing/extra keys for AESSTrainingSettings, {e}")
            elif isinstance(value, AESSTrainingSettings):
                self._training_settings = value
        else:
            raise TypeError(
                f"Training settings must be a dict or None or type AESSTrainingSettings, {type(value)} is passed."
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
        ss_decoder_output = transpose(self.ss_decoder(encoded), 1, 2).reshape(-1, self.ds)
        return devectorized, ss_decoder_output, encoded

    def save(self, model_dir, epoch):
        super(AutoencoderSSDecoder, self).save(model_dir, epoch)
        torch_save(self.ss_decoder, model_dir / f"ss_decoder_{epoch}.m")

    def load(self, model_dir, model_id):
        super(AutoencoderSSDecoder, self).load(model_dir, model_id)
        self.ss_decoder = torch_load(
            model_dir / f"ss_decoder_{model_id}.m", map_location=get_map_location()
        )

    def initialize_training_components(self):
        super(AutoencoderSSDecoder, self).initialize_training_components()
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
        if self.focus is not None:
            if self.focus == "ss_decoder":
                self.focused_optimizer = optim.SGD(
                    [
                        {"params": self.ss_decoder.parameters()},
                    ],
                    lr=self._modular_training_settings.focused.lr,
                )

    def clip_ss_decoder_gradients(self):
        # gradient clipping:
        clip_grad_value_(self.vectorizer.parameters(), clip_value=self.g_clip)
        clip_grad_value_(self.encoder.parameters(), clip_value=self.g_clip)
        clip_grad_value_(self.ss_decoder.parameters(), clip_value=self.g_clip)

    def train_ss_decoder(self, one_hot_input, target_vals_ss):
        # train encoder_SS_decoder
        self.ss_decoder_optimizer.zero_grad()
        ss_decoder_output = self.forward_ss_decoder(one_hot_input)
        ss_decoder_loss = self.criterion_NLLLoss(ss_decoder_output, target_vals_ss.reshape((-1,)))
        ss_decoder_loss.backward()
        self.clip_ss_decoder_gradients()
        self.ss_decoder_optimizer.step()
        self._training_settings.ss_decoder.lr = self.ss_decoder_lr_scheduler.get_last_lr()
        self.ss_decoder_lr_scheduler.step(ss_decoder_loss.item())
        self.log("ss_decoder_loss", ss_decoder_loss.item())
        self.log("ss_decoder_LR", self.training_settings.ss_decoder.lr)

    def train_focused(self, **kwargs):
        self.focused_optimizer.zero_grad()
        loss = None
        if self.focus in ["vectorizer", "encoder", "decoder", "devectorizer"]:
            loss = self.autoencoder_focused(**kwargs)
        elif self.focus == "ss_decoder":
            loss = self.ss_decoder_focused(**kwargs)
        if loss is not None:
            self.focused_optimizer.step()
            self._modular_training_settings.focused.lr = self.focused_lr_scheduler.get_last_lr()
            self.focused_lr_scheduler.step(loss.item())
            self.log(f"focused_{self.focus}_LR", self.focused_lr_scheduler.get_last_lr())

    def ss_decoder_focused(self, **kwargs):
        one_hot_input = kwargs["one_hot_input"]
        target_vals_ss = kwargs["target_vals_ss"]
        input_keys = kwargs["input_keys"]
        if "S" in input_keys and "C" not in input_keys:
            ss_decoder_output = self.forward_ss_decoder(one_hot_input)
            loss = self.criterion_NLLLoss(ss_decoder_output, target_vals_ss.reshape((-1,)))
            loss.backward()
            self.clip_ss_decoder_gradients()
            return loss
        return None

    def train_one_batch(self, input_vals, input_noise=0.0, device=None, input_keys="AS-"):
        if input_vals is not None:
            input_ndx, target_vals_ss, _, one_hot_input = self.transform_input(
                input_vals, device, input_noise=input_noise, input_keys=input_keys
            )
            self.train_reconstructor(one_hot_input, input_ndx)
            if "S" in input_keys:
                self.train_ss_decoder(one_hot_input, target_vals_ss)
            if self.focus is not None:
                self.train_focused(
                    one_hot_input=one_hot_input,
                    input_ndx=input_ndx,
                    target_vals_cl=target_vals_ss,
                    input_keys=input_keys,
                )

    @staticmethod
    def assert_input_type(input_vals):
        assert isinstance(input_vals, Dict), "AESS requires a dict as input_vals"

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
                input_vals["cl"], input_noise=input_noise, device=device, input_keys="A-"
            )
        if "ss" in input_vals.keys():
            self.train_one_batch(
                input_vals["ss"], input_noise=input_noise, device=device, input_keys="AS"
            )
        if "clss" in input_vals.keys():
            self.train_one_batch(
                input_vals["clss"], input_noise=input_noise, device=device, input_keys="AS-"
            )

    def test_ss_decoder(self, ss_decoder_output, target_vals_ss, device):
        ss_decoder_loss = self.criterion_NLLLoss(ss_decoder_output, target_vals_ss.reshape((-1,)))
        # ss_decoder acc
        ss_decoder_accuracy = reconstructor_acc(ss_decoder_output, target_vals_ss)
        consensus_ss_acc, _ = consensus_acc(target_vals_ss, ss_decoder_output, self.w, device)
        self.log("test_ss_decoder_loss", ss_decoder_loss.item())
        self.log("test_ss_decoder_accuracy", ss_decoder_accuracy.item())
        self.log("test_consensus_ss_accuracy", consensus_ss_acc)

    def test_one_batch(self, input_vals, device, input_keys="AS-"):
        if input_vals is not None:
            input_ndx, target_vals_ss, _, one_hot_input = self.transform_input(
                input_vals, device, input_keys=input_keys
            )
            reconstructor_output, ss_decoder_output, encoded_output = self.forward_test(
                one_hot_input
            )
            self.test_reconstructor(reconstructor_output, input_ndx, device)
            # test for continuity
            self.test_continuity(encoded_output)
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
                self.test_one_batch(input_vals["cl"], device, input_keys="A-")
            if "ss" in input_vals.keys():
                self.test_one_batch(input_vals["ss"], device, input_keys="AS")
            if "clss" in input_vals.keys():
                self.test_one_batch(input_vals["clss"], device, input_keys="AS-")

    def eval_one_batch(self, input_vals, device, input_keys="A--", embed_only=False):
        if input_vals is not None:
            _, _, _, one_hot_input = self.transform_input(input_vals, device, input_keys=input_keys)
            if embed_only:
                encoded_output = self.forward_embed(one_hot_input)
                return {"embedding": encoded_output}
            else:
                reconstructor_output, ss_decoder_output, encoded_output = self.forward_test(
                    one_hot_input
                )
                return {
                    "reconstructor_output": output_to_ndx(reconstructor_output, self.w),
                    "ss_decoder_output": output_to_ndx(ss_decoder_output, self.w),
                    "embedding": encoded_output,
                }
