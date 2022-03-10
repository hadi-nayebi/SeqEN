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
from torch.nn import Module, MSELoss, NLLLoss
from torch.nn.functional import one_hot, unfold
from torch.nn.utils import clip_grad_value_

from SeqEN2.autoencoder.utils import CustomLRScheduler, LayerMaker
from SeqEN2.model.data_loader import read_json, write_json
from SeqEN2.utils.custom_dataclasses import AETrainingSettings
from SeqEN2.utils.seq_tools import consensus_acc
from SeqEN2.utils.utils import get_map_location


# class for AE
class Autoencoder(Module):
    aa_keys = "WYFMILVAGPSTCEDQNHRK*"  # amino acids class labels
    ss_keys = "CSTIGHBE*"
    d0 = 21  # amino acids class size
    g_clip = 1.0

    def __init__(self, d1, dn, w, arch):
        super(Autoencoder, self).__init__()
        # common attr
        self.d1 = d1
        self.dn = dn
        self.w = w
        self.arch = arch
        # Modules
        self.vectorizer = LayerMaker().make(self.arch.vectorizer)
        self.encoder = LayerMaker().make(self.arch.encoder)
        self.decoder = LayerMaker().make(self.arch.decoder)
        self.devectorizer = LayerMaker().make(self.arch.devectorizer)
        # training components
        self._training_settings = AETrainingSettings()
        # define customized optimizers
        self.reconstructor_optimizer = None
        self.reconstructor_lr_scheduler = None
        self.ignore_continuity = False
        self.continuity_optimizer = None
        self.continuity_lr_scheduler = None
        # Loss functions
        self.criterion_NLLLoss = NLLLoss()
        self.criterion_MSELoss = MSELoss()
        # logger
        self.logs = {}

    @property
    def training_settings(self) -> AETrainingSettings:
        return self._training_settings

    @training_settings.setter
    def training_settings(self, value=None) -> None:
        if isinstance(value, Dict) or value is None or isinstance(value, AETrainingSettings):
            if isinstance(value, Dict):
                try:
                    self._training_settings = AETrainingSettings(**value)
                except TypeError as e:
                    raise KeyError(f"missing/extra keys for AETrainingSettings, {e}")
            elif isinstance(value, AETrainingSettings):
                self._training_settings = value
        else:
            raise TypeError(
                f"Training settings must be a dict or None or type AETrainingSettings, {type(value)} is passed."
            )

    def save_training_settings(self, train_dir):
        write_json(
            self.training_settings.to_dict(),
            str(train_dir / "training_settings.json"),
        )

    def update_training_settings(self, train_dir):
        new_training_setting_path = train_dir / "update_training_settings.json"
        if new_training_setting_path.exists():
            new_training_setting = read_json(str(new_training_setting_path))
            if new_training_setting["apply"]:
                new_training_setting_dict = {}
                for key, item in self.training_settings.to_dict().items():
                    if item["lr"] < new_training_setting[key]["lr"]:
                        new_training_setting_dict[key] = new_training_setting[key]["params"]
                    else:
                        new_training_setting_dict[key] = item
                self.initialize_for_training(training_settings=new_training_setting_dict)
                new_training_setting["apply"] = False
                write_json(new_training_setting, new_training_setting_path)

    def forward_encoder_decoder(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        decoded = transpose(self.decoder(encoded), 1, 2).reshape(-1, self.d1)
        devectorized = self.devectorizer(decoded)
        return devectorized

    def forward_test(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        decoded = transpose(self.decoder(encoded), 1, 2).reshape(-1, self.d1)
        devectorized = self.devectorizer(decoded)
        return devectorized, encoded

    def forward_embed(self, one_hot_input):
        vectorized = self.vectorizer(one_hot_input.reshape((-1, self.d0)))
        encoded = self.encoder(transpose(vectorized.reshape(-1, self.w, self.d1), 1, 2))
        return encoded

    def save(self, model_dir, epoch):
        torch_save(self.vectorizer, model_dir / f"vectorizer_{epoch}.m")
        torch_save(self.encoder, model_dir / f"encoder_{epoch}.m")
        torch_save(self.decoder, model_dir / f"decoder_{epoch}.m")
        torch_save(self.devectorizer, model_dir / f"devectorizer_{epoch}.m")

    def load(self, model_dir, model_id):
        self.vectorizer = torch_load(
            model_dir / f"vectorizer_{model_id}.m", map_location=get_map_location()
        )
        self.encoder = torch_load(
            model_dir / f"encoder_{model_id}.m", map_location=get_map_location()
        )
        self.decoder = torch_load(
            model_dir / f"decoder_{model_id}.m", map_location=get_map_location()
        )
        self.devectorizer = torch_load(
            model_dir / f"devectorizer_{model_id}.m", map_location=get_map_location()
        )

    def initialize_training_components(self):
        # define customized optimizers
        self.reconstructor_optimizer = optim.SGD(
            [
                {"params": self.vectorizer.parameters()},
                {"params": self.encoder.parameters()},
                {"params": self.decoder.parameters()},
                {"params": self.devectorizer.parameters()},
            ],
            lr=self._training_settings.reconstructor.lr,
        )
        self.reconstructor_lr_scheduler = CustomLRScheduler(
            self.reconstructor_optimizer,
            factor=self._training_settings.reconstructor.factor,
            patience=self._training_settings.reconstructor.patience,
            min_lr=self._training_settings.reconstructor.min_lr,
        )
        # define customized optimizers
        self.continuity_optimizer = optim.SGD(
            [
                {"params": self.vectorizer.parameters()},
                {"params": self.encoder.parameters()},
            ],
            lr=self._training_settings.continuity.lr,
        )
        self.continuity_lr_scheduler = CustomLRScheduler(
            self.continuity_optimizer,
            factor=self._training_settings.continuity.factor,
            patience=self._training_settings.continuity.patience,
            min_lr=self._training_settings.continuity.min_lr,
        )

    def initialize_for_training(self, training_settings=None):
        self.training_settings = training_settings
        self.initialize_training_components()

    def transform_input(self, input_vals, device, input_noise=0.0, input_keys="A--"):
        # input_keys = "A--" : sequence, "AC-" sequence:class, "AS-" sequence:ss, "ACS" seq:class:ss, "
        # scans by sliding window of w
        assert isinstance(input_vals, Tensor), f"expected Tensor type, received {type(input_vals)}"
        kernel_size = (input_vals.shape[1], self.w)
        input_vals = unfold(input_vals.float().T[None, None, :, :], kernel_size=kernel_size)[0].T
        input_ndx = input_vals[:, : self.w].long()
        target_vals_ss = None
        target_vals_cl = None
        if len(input_keys) == 2:
            if input_keys[1] == "S":
                target_vals_ss = input_vals[:, self.w :].long()
            elif input_keys[1] == "C":
                target_vals_cl = input_vals[:, self.w :].mean(axis=1).reshape((-1, 1))
                target_vals_cl = cat((target_vals_cl, 1 - target_vals_cl), 1).float()
        elif len(input_keys) == 3:
            if input_keys[1] == "S":
                target_vals_ss = input_vals[:, self.w : -self.w].long()
            elif input_keys[2] == "S":
                target_vals_ss = input_vals[:, -self.w :].long()
            if input_keys[1] == "C":
                target_vals_cl = input_vals[:, self.w : -self.w].mean(axis=1).reshape((-1, 1))
                target_vals_cl = cat((target_vals_cl, 1 - target_vals_cl), 1).float()
            elif input_keys[2] == "C":
                target_vals_cl = input_vals[:, -self.w :].mean(axis=1).reshape((-1, 1))
                target_vals_cl = cat((target_vals_cl, 1 - target_vals_cl), 1).float()
        # one-hot vec input
        one_hot_input = one_hot(input_ndx, num_classes=self.d0) * 1.0
        if input_noise > 0.0:
            ndx = randperm(self.w)
            size = list(one_hot_input.shape)
            size[-1] = 1
            p = tensor(choice([1, 0], p=[input_noise, 1 - input_noise], size=size)).to(device)
            one_hot_input = (one_hot_input[:, ndx, :] * p) + (one_hot_input * (1 - p))
        return input_ndx, target_vals_ss, target_vals_cl, one_hot_input

    def log(self, key, value):
        if key in self.logs.keys():
            self.logs[key].append(value)
        else:
            self.logs[key] = [value]

    def reset_log(self):
        self.logs = {}

    def clip_reconstructor_gradients(self):
        # gradient clipping:
        clip_grad_value_(self.vectorizer.parameters(), clip_value=self.g_clip)
        clip_grad_value_(self.encoder.parameters(), clip_value=self.g_clip)
        clip_grad_value_(self.decoder.parameters(), clip_value=self.g_clip)
        clip_grad_value_(self.devectorizer.parameters(), clip_value=self.g_clip)

    def clip_continuity_gradients(self):
        # gradient clipping:
        clip_grad_value_(self.vectorizer.parameters(), clip_value=self.g_clip)
        clip_grad_value_(self.encoder.parameters(), clip_value=self.g_clip)

    def train_reconstructor(self, one_hot_input, input_ndx):
        # train encoder_decoder
        self.reconstructor_optimizer.zero_grad()
        reconstructor_output = self.forward_encoder_decoder(one_hot_input)
        reconstructor_loss = self.criterion_NLLLoss(reconstructor_output, input_ndx.reshape((-1,)))
        reconstructor_loss.backward()
        self.clip_reconstructor_gradients()
        self.reconstructor_optimizer.step()
        self.log("reconstructor_loss", reconstructor_loss.item())
        self.log("reconstructor_LR", self.reconstructor_lr_scheduler.get_last_lr())
        self._training_settings.reconstructor.lr = self.reconstructor_lr_scheduler.get_last_lr()
        self.reconstructor_lr_scheduler.step(reconstructor_loss.item())

    def train_continuity(self, one_hot_input):
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
            self.clip_continuity_gradients()
            self.continuity_optimizer.step()
            self.log("continuity_loss", continuity_loss.item())
            self.log("continuity_LR", self.continuity_lr_scheduler.get_last_lr())
            self._training_settings.continuity.lr = self.continuity_lr_scheduler.get_last_lr()
            self.continuity_lr_scheduler.step(continuity_loss.item())

    def train_one_batch(self, input_vals, input_noise=0.0, device=None, input_keys="A--"):
        if input_vals is not None:
            input_ndx, _, _, one_hot_input = self.transform_input(
                input_vals, device, input_noise=input_noise, input_keys=input_keys
            )
            self.train_reconstructor(one_hot_input, input_ndx)
            # train for continuity
            self.train_continuity(one_hot_input)

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
                input_vals["ss"], input_noise=input_noise, device=device, input_keys="A-"
            )
        if "clss" in input_vals.keys():
            self.train_one_batch(
                input_vals["clss"], input_noise=input_noise, device=device, input_keys="A--"
            )

    def test_reconstructor(self, reconstructor_output, input_ndx, device):
        reconstructor_loss = self.criterion_NLLLoss(reconstructor_output, input_ndx.reshape((-1,)))
        # reconstructor acc
        reconstructor_ndx = argmax(reconstructor_output, dim=1)
        reconstructor_accuracy = (
            torch_sum(reconstructor_ndx == input_ndx.reshape((-1,))) / reconstructor_ndx.shape[0]
        )
        consensus_seq_acc, _ = consensus_acc(
            input_ndx, reconstructor_ndx.reshape((-1, self.w)), device
        )
        # reconstruction_loss, discriminator_loss, classifier_loss
        self.log("test_reconstructor_loss", reconstructor_loss.item())
        self.log("test_reconstructor_accuracy", reconstructor_accuracy.item())
        self.log("test_consensus_accuracy", consensus_seq_acc)

    def test_continuity(self, encoded_output):
        if not self.ignore_continuity:
            continuity_loss_r = self.criterion_MSELoss(
                encoded_output, cat((encoded_output[1:], encoded_output[-1].unsqueeze(0)), 0)
            )
            continuity_loss_l = self.criterion_MSELoss(
                encoded_output, cat((encoded_output[0].unsqueeze(0), encoded_output[:-1]), 0)
            )
            continuity_loss = continuity_loss_r + continuity_loss_l
            self.log("test_continuity_loss", continuity_loss.item())

    def test_one_batch(self, input_vals, device, input_keys="A--"):
        if input_vals is not None:
            input_ndx, _, _, one_hot_input = self.transform_input(
                input_vals, device, input_keys=input_keys
            )
            reconstructor_output, encoded_output = self.forward_test(one_hot_input)
            self.test_reconstructor(reconstructor_output, input_ndx, device)
            # test for continuity
            self.test_continuity(encoded_output)

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
                self.test_one_batch(input_vals["ss"], device, input_keys="A-")
            if "clss" in input_vals.keys():
                self.test_one_batch(input_vals["clss"], device, input_keys="A--")

    @staticmethod
    def assert_input_type(input_vals):
        assert isinstance(input_vals, Dict), "AE requires a dict as input_vals"

    def eval_one_batch(self, input_vals, device, input_keys="A--", embed_only=False):
        if input_vals is not None:
            _, _, _, one_hot_input = self.transform_input(input_vals, device, input_keys=input_keys)
            if embed_only:
                encoded_output = self.forward_embed(one_hot_input)
                return {"embedding": encoded_output}
            else:
                reconstructor_output, encoded_output = self.forward_test(one_hot_input)
                return {
                    "reconstructor_output": reconstructor_output,
                    "embedding": encoded_output,
                }

    def eval_batch(self, input_vals, device, embed_only=False):
        self.assert_input_type(input_vals)
        assert len(input_vals) == 1, "more than one item in input_vals for eval"
        self.eval()
        with no_grad():
            if "cl" in input_vals.keys():
                return self.eval_one_batch(
                    input_vals["cl"], device, input_keys="A-", embed_only=embed_only
                )
            elif "ss" in input_vals.keys():
                return self.eval_one_batch(
                    input_vals["ss"], device, input_keys="A-", embed_only=embed_only
                )
            elif "clss" in input_vals.keys():
                return self.eval_one_batch(
                    input_vals["clss"], device, input_keys="A--", embed_only=embed_only
                )
