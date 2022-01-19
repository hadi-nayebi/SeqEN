"""Define DataLoader class and related I/O functions."""

import gzip
import json
from os.path import dirname
from pathlib import Path
from typing import Any

from numpy import array, ndarray
from numpy.random import randint
from torch import cat, tensor


class NumpyEncoder(json.JSONEncoder):
    """Enables JSON too encode numpy nd-arrays."""

    def default(self, obj) -> Any:
        if isinstance(obj, ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def read_fasta(filename) -> dict:
    """Read fasta files and return a dict."""
    data_dict = {}
    with open(filename, "r") as file:
        for line in file.readlines():
            if line.startswith(">"):
                key = line.strip()[1:]
                data_dict[key] = ""
            else:
                data_dict[key] += line.strip()
    return data_dict


def read_json(filename) -> dict:
    """Read json files and return a dict. (.json, .json.gz)"""
    if filename.endswith(".json.gz"):
        with gzip.open(filename, "r") as file:
            json_bytes = file.read()
            json_str = json_bytes.decode("utf-8")
            return json.loads(json_str)
    elif filename.endswith(".json"):
        with open(filename, "r") as file:
            return json.load(file)
    else:
        raise IOError("File format must be .gz or .json.gz")


def write_json(data_dict, filename, encoder=None) -> None:
    """Write json file from a dict, encoding numpy arrays. (.json, .json.gz)"""
    if encoder == "numpy":
        encoder = NumpyEncoder
    elif encoder is None:
        pass
    else:
        raise NotImplemented(f"No encoding implemented for {encoder}")
    json_str = json.dumps(data_dict, cls=encoder) + "\n"
    if filename.endswith(".json.gz"):
        with gzip.open(filename, "w") as file:
            json_bytes = json_str.encode("utf-8")
            file.write(json_bytes)
    elif filename.endswith(".json"):
        with open(filename, "w") as file:
            file.write(json_str)
    else:
        raise IOError("File format must be .gz or .json.gz")


def join(items) -> tensor:
    return cat(items, 0)


def to_tensor(data, key, device) -> tensor:
    output = None
    metadata = {"name": key}
    for i, (key, value) in enumerate(data.items()):
        if output is None:
            output = tensor(value, device=device).reshape((-1, 1))
            metadata[f"{i}"] = {"name": key, "shape": len(value)}
        else:
            output = cat((output, tensor(value, device=device).reshape((-1, 1))), 1)
            metadata[f"{i}"] = {"name": key, "shape": len(value)}
    assert output is not None
    return output, metadata


class DataLoader(object):
    """DataLoader maintains train/test data for training/testing model."""

    root = Path(dirname(__file__)).parent.parent

    def __init__(self) -> None:
        self._train_data = None
        self._test_data = None
        self.test_data_keys = None

    @property
    def train_data(self) -> dict:
        return self._train_data

    @property
    def test_data(self) -> dict:
        return self._test_data

    def load_test_data(self, dataset, device) -> None:
        filename = self.root / "data" / f"{dataset}_test.json.gz"
        self._test_data = read_json(str(filename))
        # to tensor, metadata
        for key in self._test_data.keys():
            self._test_data[key] = to_tensor(self._test_data[key], key, device)
        self.test_data_keys = list(self._test_data.keys())

    def load_train_data(self, dataset, device) -> None:
        filename = self.root / "data" / f"{dataset}_train.json.gz"
        self._train_data = read_json(str(filename))
        # to tensor, metadata
        for key in self._train_data.keys():
            self._train_data[key] = to_tensor(self._train_data[key], key, device=device)

    def get_train_batch(self, batch_size=128, max_size=None) -> array:
        num_batch = len(self._train_data) // batch_size
        if max_size is not None:
            assert max_size < len(self._test_data), "size is bigger that test data items."
            num_batch = max_size // batch_size
        keys = list(self._train_data.keys())
        for i in range(num_batch + 1):
            yield join(
                [self._train_data[key][0] for key in keys[i * batch_size : (i + 1) * batch_size]]
            )

    def get_test_batch(self, batch_size=1) -> (tensor, dict):
        ndx = randint(0, len(self.test_data_keys), batch_size)
        for i in ndx:
            key = self.test_data_keys[i]
            yield self._test_data[key]
