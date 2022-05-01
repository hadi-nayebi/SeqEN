#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from typing import Dict

from torch import Tensor
from torch import load as torch_load
from torch import no_grad, optim
from torch import save as torch_save
from torch import transpose
from torch.nn import Module, MSELoss, NLLLoss
from torch.nn.functional import one_hot
from torch.nn.utils import clip_grad_value_

from SeqEN2.autoencoder.utils import CustomLRScheduler, FocusedLRScheduler, LayerMaker
from SeqEN2.model.data_loader import read_json, write_json
from SeqEN2.utils.custom_dataclasses import AETrainingSettings, ModularTrainingSettings
from SeqEN2.utils.seq_tools import (
    add_noise,
    consensus_acc,
    continuity_target_left,
    continuity_target_right,
    output_to_ndx,
    reconstructor_acc,
    slide_window,
    split_input_vals,
)
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
        self._modular_training_settings = ModularTrainingSettings()
        # define customized optimizers
        self.reconstructor_optimizer = None
        self.reconstructor_lr_scheduler = None
        self.ignore_continuity = False
        self.continuity_optimizer = None
        self.continuity_lr_scheduler = None
        # by module
        self.focus = None
        self.focused_optimizer = None
        self.focused_lr_scheduler = None
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

    @property
    def modular_training_settings(self) -> ModularTrainingSettings:
        return self._modular_training_settings

    @modular_training_settings.setter
    def modular_training_settings(self, value=None) -> None:
        if isinstance(value, Dict) or value is None or isinstance(value, ModularTrainingSettings):
            if isinstance(value, Dict):
                try:
                    self._modular_training_settings = ModularTrainingSettings(**value)
                except TypeError as e:
                    raise KeyError(f"missing/extra keys for ModularTrainingSettings, {e}")
            elif isinstance(value, ModularTrainingSettings):
                self._modular_training_settings = value
        else:
            raise TypeError(
                f"Training settings must be a dict or None or type ModularTrainingSettings, {type(value)} is passed."
            )

    def save_training_settings(self, train_dir):
        write_json(
            self.training_settings.to_dict(),
            str(train_dir / "training_settings.json"),
        )

    def save_modular_training_settings(self, train_dir):
        write_json(
            self.modular_training_settings.to_dict(),
            str(train_dir / "modular_training_settings.json"),
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

    def update_modular_training_settings(self, train_dir):
        new_modular_training_setting_path = train_dir / "update_modular_training_settings.json"
        if new_modular_training_setting_path.exists():
            new_modular_training_setting = read_json(str(new_modular_training_setting_path))
            if new_modular_training_setting["apply"]:
                new_modular_training_setting_dict = {}
                for key, item in self.modular_training_settings.to_dict().items():
                    new_modular_training_setting_dict[key] = new_modular_training_setting[key][
                        "params"
                    ]
                self.initialize_for_training(
                    modular_training_settings=new_modular_training_setting_dict
                )
                new_modular_training_setting["apply"] = False
                write_json(new_modular_training_setting, new_modular_training_setting_path)

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
        if self.focus is not None:
            if self.focus == "vectorizer":
                self.focused_optimizer = optim.SGD(
                    [
                        {"params": self.vectorizer.parameters()},
                    ],
                    lr=self._modular_training_settings.focused.lr,
                )
            elif self.focus == "encoder":
                self.focused_optimizer = optim.SGD(
                    [
                        {"params": self.encoder.parameters()},
                    ],
                    lr=self._modular_training_settings.focused.lr,
                )
            elif self.focus == "decoder":
                self.focused_optimizer = optim.SGD(
                    [
                        {"params": self.decoder.parameters()},
                    ],
                    lr=self._modular_training_settings.focused.lr,
                )
            elif self.focus == "devectorizer":
                self.focused_optimizer = optim.SGD(
                    [
                        {"params": self.devectorizer.parameters()},
                    ],
                    lr=self._modular_training_settings.focused.lr,
                )

    def setup_focused_lr_scheduler(self):
        if self.focused_optimizer is not None:
            self.focused_lr_scheduler = FocusedLRScheduler(
                self.focused_optimizer,
                self._modular_training_settings.focused.lr,
                factor=self._modular_training_settings.focused.factor,
                patience=self._modular_training_settings.focused.patience,
                min_lr=self._modular_training_settings.focused.min_lr,
                max_lr=self._modular_training_settings.focused.max_lr,
                max_loss_change=self._modular_training_settings.focused.max_loss_change,
                min_loss_change=self._modular_training_settings.focused.min_loss_change,
            )

    def initialize_for_training(self, training_settings=None, modular_training_settings=None):
        self.training_settings = training_settings
        self.modular_training_settings = modular_training_settings
        self.initialize_training_components()
        self.setup_focused_lr_scheduler()

    def transform_input(self, input_vals, device, input_noise=0.0, input_keys="A--"):
        # input_keys = "A--" : sequence, "AC-" sequence:class, "AS-" sequence:ss, "ACS" seq:class:ss, "
        # scans by sliding window of w
        assert isinstance(input_vals, Tensor), f"expected Tensor type, received {type(input_vals)}"
        input_vals = slide_window(input_vals, self.w)
        input_ndx = input_vals[:, : self.w].long()
        target_vals_ss, target_vals_cl = split_input_vals(input_vals, input_keys, self.w)
        # one-hot vec input
        one_hot_input = one_hot(input_ndx, num_classes=self.d0) * 1.0
        if input_noise > 0.0:
            one_hot_input = add_noise(one_hot_input, input_noise, device)
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
                encoded_output, continuity_target_right(encoded_output)
            )
            continuity_loss_l = self.criterion_MSELoss(
                encoded_output, continuity_target_left(encoded_output)
            )
            continuity_loss = continuity_loss_r + continuity_loss_l
            continuity_loss.backward()
            self.clip_continuity_gradients()
            self.continuity_optimizer.step()
            self.log("continuity_loss", continuity_loss.item())
            self.log("continuity_LR", self.continuity_lr_scheduler.get_last_lr())
            self._training_settings.continuity.lr = self.continuity_lr_scheduler.get_last_lr()
            self.continuity_lr_scheduler.step(continuity_loss.item())

    def train_focused(self, **kwargs):
        self.focused_optimizer.zero_grad()
        loss = None
        if self.focus in ["vectorizer", "encoder", "decoder", "devectorizer"]:
            loss = self.autoencoder_focused(**kwargs)
        self.focused_optimizer.step()
        self._modular_training_settings.focused.lr = self.focused_lr_scheduler.get_last_lr()
        self.focused_lr_scheduler.step(loss.item())
        self.log(f"focused_{self.focus}_LR", self.focused_lr_scheduler.get_last_lr())

    def autoencoder_focused(self, **kwargs):
        one_hot_input = kwargs["one_hot_input"]
        input_ndx = kwargs["input_ndx"]
        reconstructor_output = self.forward_encoder_decoder(one_hot_input)
        loss = self.criterion_NLLLoss(reconstructor_output, input_ndx.reshape((-1,)))
        loss.backward()
        self.clip_reconstructor_gradients()
        return loss

    def train_one_batch(self, input_vals, input_noise=0.0, device=None, input_keys="A--"):
        if input_vals is not None:
            input_ndx, _, _, one_hot_input = self.transform_input(
                input_vals, device, input_noise=input_noise, input_keys=input_keys
            )
            self.train_reconstructor(one_hot_input, input_ndx)
            # train for continuity
            self.train_continuity(one_hot_input)
            if self.focus is not None:
                self.train_focused(one_hot_input=one_hot_input, input_ndx=input_ndx)

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
        reconstructor_accuracy = reconstructor_acc(reconstructor_output, input_ndx)
        consensus_seq_acc, _ = consensus_acc(input_ndx, reconstructor_output, self.w, device)
        # reconstruction_loss, discriminator_loss, classifier_loss
        self.log("test_reconstructor_loss", reconstructor_loss.item())
        self.log("test_reconstructor_accuracy", reconstructor_accuracy.item())
        self.log("test_consensus_accuracy", consensus_seq_acc)

    def test_continuity(self, encoded_output):
        if not self.ignore_continuity:
            continuity_loss_r = self.criterion_MSELoss(
                encoded_output, continuity_target_right(encoded_output)
            )
            continuity_loss_l = self.criterion_MSELoss(
                encoded_output, continuity_target_left(encoded_output)
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
                    "reconstructor_output": output_to_ndx(reconstructor_output, self.w),
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
