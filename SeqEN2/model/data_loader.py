"""Define DataLoader class and related I/O functions."""

import gzip
import json
from os.path import dirname
from pathlib import Path
from typing import Any

from numpy import array, concatenate, int8, ndarray


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


def join(items) -> ndarray:
    return concatenate(items, axis=0)


def to_array(data) -> ndarray:
    output = array(data["ndx"]).reshape((-1, 1))
    return concatenate([output, array(data["ACT_p"]).reshape((-1, 1))], axis=1)


class DataLoader(object):
    """DataLoader maintains train/test data for training/testing model."""

    root = Path(dirname(__file__)).parent.parent

    def __init__(self) -> None:
        self._train_data = None
        self._test_data = None

    @property
    def train_data(self) -> dict:
        return self._train_data

    @property
    def test_data(self) -> dict:
        return self._test_data

    def load_test_data(self, dataset) -> None:
        filename = self.root / "data" / f"{dataset}_test.json.gz"
        self._test_data = read_json(str(filename))
        # to numpy array
        for key in self._test_data.keys():
            self._test_data[key] = to_array(self._test_data[key])

    def load_train_data(self, dataset) -> None:
        filename = self.root / "data" / f"{dataset}_train.json.gz"
        self._train_data = read_json(str(filename))
        # to numpy array
        for key in self._train_data.keys():
            self._train_data[key] = to_array(self._train_data[key])

    def get_train_batch(self, batch_size=128) -> array:
        num_batch = len(self._train_data) // batch_size
        keys = list(self._train_data.keys())
        for i in range(num_batch + 1):
            batch = join(
                [self._train_data[key] for key in keys[i * batch_size : (i + 1) * batch_size]]
            )
            yield batch

    def get_test_batch(self, batch_size=1) -> array:
        for key in list(self._test_data.keys())[:batch_size]:
            yield self._test_data[key]
