#!/usr/bin/env python
# coding: utf-8

"""Unit test DataLoader."""

from typing import Dict
from unittest import TestCase
from unittest import main as unittest_main

from torch import Tensor, cuda, device

from SeqEN2.model.data_loader import DataLoader


class TestDataLoader(TestCase):
    """Test items for DataLoader class"""

    device = device("cuda" if cuda.is_available() else "cpu")

    def test_load_test_data(self):
        data_loader = DataLoader()
        # testing for seq:p_ACT data
        dataset_name = "KeggSeq_ndx_wpACT_100"
        data_loader.load_test_data(dataset_name, device=self.device)
        assert len(data_loader.test_data) == 100, "loaded data size do not match"
        first_item = list(data_loader.test_data.values())[0]
        assert len(first_item) == 2, "DataLoader must return two items, data  and metadata "
        assert isinstance(first_item[0], Tensor), "The first item from DataLoader must be a tensor"
        assert first_item[0].shape[1] == 2, "The value shape from DataLoader must be (-1, 2)"
        assert isinstance(first_item[1], Dict), "The second item from DataLoader must be a dict"
        assert list(first_item[1].keys()) == ["name", "0", "1"], "problem with metadata"
        name, ndx, act_p = first_item[1].values()
        assert isinstance(name, str), "name in metadata must be a string"
        assert isinstance(ndx, Dict), "ndx in metadata must be a dict"
        assert ndx["name"] == "ndx", "name of second item in metadata must be ndx"
        assert isinstance(act_p, Dict), "act_p in metadata must be a dict"
        assert act_p["name"] == "ACT_p", "name of second item in metadata must be ACT_p"
        assert ndx["shape"] == act_p["shape"], "ndx and ACT_p shape do not match"

        # testing for seq:ss data
        dataset_name = "pdb_ss_ndx_100"
        data_loader.load_test_data(dataset_name, device=self.device)
        assert len(data_loader.test_data) == 100, "loaded data size do not match"
        first_item = list(data_loader.test_data.values())[0]
        assert len(first_item) == 2, "DataLoader must return two items, data  and metadata "
        assert isinstance(first_item[0], Tensor), "The first item from DataLoader must be a tensor"
        assert first_item[0].shape[1] == 2, "The value shape from DataLoader must be (-1, 2)"
        assert isinstance(first_item[1], Dict), "The second item from DataLoader must be a dict"
        assert list(first_item[1].keys()) == ["name", "0", "1"], "problem with metadata"
        name, ndx, ss = first_item[1].values()
        assert isinstance(name, str), "name in metadata must be a string"
        assert isinstance(ndx, Dict), "ndx in metadata must be a dict"
        assert ndx["name"] == "ndx", "name of second item in metadata must be ndx"
        assert isinstance(act_p, Dict), "ss in metadata must be a dict"
        assert ss["name"] == "ss", "name of second item in metadata must be ss"
        assert ndx["shape"] == ss["shape"], "ndx and ss shape do not match"

    # def test_load_train_data(self):
    #     dataset_name = "KeggSeq_ndx_wpACT_100"
    #     data_loader = DataLoader()
    #     data_loader.load_train_data(dataset_name)
    #     assert isinstance([item for item in data_loader.train_data.values()][0], ndarray)
    #
    # def test_get_test_batch(self):
    #     dataset_name = "KeggSeq_ndx_wpACT_100"
    #     data_loader = DataLoader()
    #     data_loader.load_test_data(dataset_name)
    #     batch_size = 5
    #     assert (
    #         len([item for item in data_loader.get_test_batch(batch_size)]) == batch_size
    #     ), "returned wrong  number of items"
    #
    # def test_get_train_batch(self):
    #     dataset_name = "KeggSeq_ndx_wpACT_100"
    #     data_loader = DataLoader()
    #     data_loader.load_train_data(dataset_name)
    #     large_batch = small_batch = 0
    #     for item in data_loader.get_train_batch(10):
    #         large_batch = item.shape
    #         break
    #     for item in data_loader.get_train_batch(1):
    #         small_batch = item.shape
    #         break
    #     assert large_batch > small_batch, "train batch fails"


if __name__ == "__main__":
    unittest_main()
