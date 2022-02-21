"""Define DataLoader class and related I/O functions."""

import gzip
import json
from os.path import dirname
from pathlib import Path
from typing import Any, Dict

from numpy import arange, array, concatenate, ndarray
from numpy.random import choice, permutation
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
    if isinstance(filename, Path):
        filename = str(filename)
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
    if isinstance(filename, Path):
        filename = str(filename)
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
    output = cat(items, 0)
    return output


def to_tensor(data, key, device) -> (tensor, Dict):
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
        self.train_data_size = None
        self.test_data_size = None

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
        self.test_data_size = len(self._test_data)

    def load_train_data(self, dataset, device) -> None:
        filename = self.root / "data" / f"{dataset}_train.json.gz"
        self._train_data = read_json(str(filename))
        # to tensor, metadata
        for key in self._train_data.keys():
            self._train_data[key] = to_tensor(self._train_data[key], key, device)
        self.train_data_size = len(self._train_data)

    def get_train_batch(self, batch_size=128, max_size=None) -> array:
        keys = permutation(list(self._train_data.keys()))
        keys_len = len(keys)
        num_batch = keys_len // batch_size
        if max_size is not None:
            num_batch = max_size // batch_size
            if max_size > keys_len:
                repeat_data_fold = (max_size // keys_len) + 1
                keys = concatenate([keys] * repeat_data_fold, axis=0)
        for i in range(num_batch):
            yield join(
                [self._train_data[key][0] for key in keys[i * batch_size : (i + 1) * batch_size]]
            )

    def get_test_batch(self, batch_size=1, test_items=None) -> (tensor, dict):
        if test_items is not None:
            for key in test_items:
                yield self._test_data[key]
        else:
            if batch_size == -1:
                batch_size = len(self.test_data_keys)
            assert batch_size <= self.test_data_size, "batch size is bigger than available data."
            ndx = choice(arange(len(self.test_data_keys)), size=batch_size, replace=False)
            for i in ndx:
                key = self.test_data_keys[i]
                yield self._test_data[key]

    def get_by_key(self, key, dataset="test") -> (tensor, dict):
        if dataset == "test":
            return self._test_data[key]
        elif dataset == "train":
            return self._train_data[key]

    def get_all(self):
        all_data = self._test_data.copy()
        all_data.update(self._train_data)
        for key in all_data.keys():
            yield all_data[key]
