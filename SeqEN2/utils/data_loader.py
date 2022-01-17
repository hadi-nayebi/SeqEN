"""Define DataLoader class and related I/O functions."""
# TODO: this will be removed, new data loader in model dir works with new data format ndx.json

import gzip
import json
from typing import Any

from numpy import array, ndarray, unique
from numpy.random import choice
from pandas import DataFrame, concat, read_csv


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
    if filename.endswith(".json.gz"):
        json_str = json.dumps(data_dict, cls=encoder) + "\n"
        json_bytes = json_str.encode("utf-8")
        with gzip.open(filename, "w") as file:
            file.write(json_bytes)
    elif filename.endswith(".json"):
        with open(filename, "w") as file:
            json.dump(data_dict, file)
    else:
        raise IOError("File format must be .gz or .json.gz")


def load_csv(filename) -> DataFrame:
    """Load one or more csv as pandas DataFrame. (.csv, .csv.gz)"""
    if isinstance(filename, list):
        return concat([load_csv(fp) for fp in filename])
    elif filename.endswith(".csv.gz") or filename.endswith(".csv"):
        return read_csv(filename, index_col=0)
    else:
        raise IOError("File (or list of files) format must be .csv or .csv.gz")


class DataLoader(object):
    """DataLoader maintains train/test data for training/testing model."""

    # TODO: DataLoader loads indexed protein data and creates sliding window on yield

    def __init__(self) -> None:
        self._train_data_files = None
        self._test_data_files = None
        self._train_data = None
        self._test_data = None
        self._test_items = None

    @property
    def train_data_files(self) -> list:
        return self._train_data_files

    @train_data_files.setter
    def train_data_files(self, train_data_files) -> None:
        self._train_data_files = train_data_files

    @property
    def test_data_files(self) -> list:
        return self._test_data_files

    @test_data_files.setter
    def test_data_files(self, test_data_files) -> None:
        self._test_data_files = test_data_files

    @property
    def train_data(self) -> DataFrame:
        return self._train_data

    @property
    def test_data(self) -> DataFrame:
        return self._test_data

    @property
    def test_items(self) -> array:
        return self._test_items

    def prepare_test_data(self) -> None:
        """Extract unique protein keys when loading test data for the first time."""
        # TODO: will be removed
        if self._test_data is None:
            self._test_data = load_csv(self._test_data_files)
            self._test_items = unique(
                ["_".join(item.split("_")[:-1]) for item in self.test_data.index]
            )

    def get_train_batch(self, batch_size=128) -> array:
        """Generator yield single batch of data as numpy nd array."""
        if self._train_data_files is not None:
            for train_data_file in self._train_data_files:
                # Loading one file at a time for memory limitation
                # TODO: improve by making it dynamic, optimize by available memory
                self._train_data = load_csv(train_data_file).sample(frac=1).reset_index(drop=True)
                num_batch = len(self._train_data) // batch_size
                for i in range(num_batch + 1):
                    yield self._train_data.iloc[i * batch_size : (i + 1) * batch_size].values
        # else: raise NotDefinedError('train data files are not defined')

    def get_test_batch(self, num_test_items=10, by_random=True) -> array:
        """Generator yield single batch of data as numpy nd array."""
        if self._train_data_files is not None:
            self.prepare_test_data()
            if num_test_items == -1:
                num_test_items = len(self._test_items)
            if by_random:
                for item in choice(self._test_items, size=num_test_items):
                    yield self._test_data[self._test_data.index.str.contains(item)].values
            else:
                for item in self._test_items[:num_test_items]:
                    yield self._test_data[self._test_data.index.str.contains(item)].values
        # else: raise NotDefinedError('test data files are not defined')
