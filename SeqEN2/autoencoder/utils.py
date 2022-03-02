#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"


from torch import optim
from torch.nn import (
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
            return Linear(layer["in"], layer["out"], bias=bias)
        elif layer["type"] == "Tanh":
            return Tanh()
        elif layer["type"] == "Sigmoid":
            return Sigmoid()
        elif layer["type"] == "ReLU":
            return ReLU()
        elif layer["type"] == "Conv1d":
            bias = layer.get("bias", True)
            padding = layer.get("padding", 0)
            return Conv1d(layer["in"], layer["out"], layer["kernel"], padding=padding, bias=bias)
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
            return ConvTranspose1d(layer["in"], layer["out"], layer["kernel"], padding=padding)


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
