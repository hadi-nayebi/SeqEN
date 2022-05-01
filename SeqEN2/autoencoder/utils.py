#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from numpy import append, array, mean
from torch import optim
from torch.nn import (
    ELU,
    Conv1d,
    ConvTranspose1d,
    Flatten,
    Linear,
    LogSoftmax,
    MaxPool1d,
    ReLU,
    Sequential,
    Sigmoid,
    Softmax,
    Tanh,
    Unflatten,
)
from torch.nn.init import kaiming_normal_, xavier_normal_


class Architecture(object):
    """
    The Architecture object provides the model arch params as a json/dict
    """

    def __init__(self, architecture):
        if isinstance(architecture, dict):
            self.architecture = architecture
        else:
            raise TypeError(f"Architecture must be of type dict. {type(architecture)} is received.")
        self.name = self.architecture["name"]
        self.type = self.architecture["type"]
        self.vectorizer = None
        self.devectorizer = None
        self.encoder = None
        self.decoder = None
        self.discriminator = None
        self.classifier = None
        self.ss_decoder = None
        self.parse_architecture()

    def parse_architecture(self):
        for key, item in self.architecture.items():
            if key == "vectorizer":
                self.vectorizer = item
            elif key == "devectorizer":
                self.devectorizer = item
            elif key == "encoder":
                self.encoder = item
            elif key == "decoder":
                self.decoder = item
            elif key == "discriminator":
                self.discriminator = item
            elif key == "classifier":
                self.classifier = item
            elif key == "ss_decoder":
                self.ss_decoder = item


class LayerMaker(object):
    """
    The LayerMaker object will host related functions to build ML models
    """

    def make(self, arch):
        layers = []
        for layer in arch:
            layers.append(self.make_layer(layer))
        return Sequential(*layers)

    @staticmethod
    def make_layer(layer):
        if layer["type"] == "Linear":
            bias = layer.get("bias", True)
            new_layer = Linear(layer["in"], layer["out"], bias=bias)
            if "init" in layer.keys():
                if layer["init"] == "xavier":
                    xavier_normal_(new_layer.weight)
                elif layer["init"] == "he":
                    kaiming_normal_(new_layer.weight)
            return new_layer
        elif layer["type"] == "Tanh":
            return Tanh()
        elif layer["type"] == "Sigmoid":
            return Sigmoid()
        elif layer["type"] == "ReLU":
            return ReLU()
        elif layer["type"] == "ELU":
            return ELU()
        elif layer["type"] == "Conv1d":
            bias = layer.get("bias", True)
            padding = layer.get("padding", 0)
            new_layer = Conv1d(
                layer["in"], layer["out"], layer["kernel"], padding=padding, bias=bias
            )
            if "init" in layer.keys():
                if layer["init"] == "xavier":
                    xavier_normal_(new_layer.weight)
                elif layer["init"] == "he":
                    kaiming_normal_(new_layer.weight)
            return new_layer
        elif layer["type"] == "LogSoftmax":
            return LogSoftmax(dim=1)
        elif layer["type"] == "Softmax":
            return Softmax(dim=1)
        elif layer["type"] == "MaxPool1d":
            return MaxPool1d(layer["kernel"])
        elif layer["type"] == "Flatten":
            return Flatten()
        elif layer["type"] == "Unflatten":
            return Unflatten(1, (layer["in"], layer["out"]))
        elif layer["type"] == "ConvTranspose1d":
            padding = layer.get("padding", 0)
            new_layer = ConvTranspose1d(layer["in"], layer["out"], layer["kernel"], padding=padding)
            if "init" in layer.keys():
                if layer["init"] == "xavier":
                    xavier_normal_(new_layer.weight)
                elif layer["init"] == "he":
                    kaiming_normal_(new_layer.weight)
            return new_layer


class CustomLRScheduler(optim.lr_scheduler.ReduceLROnPlateau):
    """
    CustomLRScheduler adds get_last_lr method to ReduceLROnPlateau class
    """

    optimizer = None

    def __init__(self, *args, **kwargs):
        super(CustomLRScheduler, self).__init__(*args, **kwargs)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr[0]


def print_shapes(inputs, module, module_name):
    print(f"input shape for {module_name}: {inputs.shape}")
    for name, unit in module.named_children():
        inputs = unit(inputs)
        print(name, inputs.shape)
    return inputs


class FocusedLRScheduler:
    def __init__(
        self,
        optimizer,
        lr,
        patience=1000,
        factor=0.99,
        min_lr=0.0001,
        max_lr=0.01,
        max_loss_change=0.1,
        min_loss_change=0.01,
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.max_loss_change = max_loss_change
        self.min_loss_change = min_loss_change

        self.loss_container = array([])
        self.loss_change = 0.0
        self.lr = array([lr])
        self.wait = 0
        self.waiting = False
        self.direction = None
        self.force = False

    def step(self, loss):
        # see if loss has changed significantly
        if self.add_new_loss(loss):
            if self.loss_change < self.max_loss_change:
                self.direction = "up"
            else:
                self.direction = "down"
                self.force = True
        self.update_waiting()
        if self.force or self.update_lr():
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.get_last_lr()
            self.force = False

    def update_lr(self):
        if not self.waiting and self.direction is not None:
            new_lr = self.lr[-1]
            if self.direction == "up":
                new_lr = self.lr[-1] * (1 + (1 - self.factor))
            elif self.direction == "down":
                new_lr = self.lr[-1] * self.factor
            new_lr = max(min(new_lr, self.max_lr), self.min_lr)
            self.lr = append(self.lr, new_lr)
            if self.lr.shape[0] > 10:
                self.lr = self.lr[-10:]
            self.waiting = True
            return True
        return False

    def get_last_lr(self):
        return self.lr[-1]

    def update_waiting(self):
        if self.waiting:
            self.wait += 1
            if self.wait == self.patience:
                self.wait = 0
                self.waiting = False

    def add_new_loss(self, loss):
        self.loss_container = append(self.loss_container, loss)
        if self.loss_container.shape[0] > self.patience:
            self.loss_container = self.loss_container[-self.patience :]
            loss_mean = mean(self.loss_container)
            self.loss_change = (loss - loss_mean) / loss_mean
            return True
        return False
