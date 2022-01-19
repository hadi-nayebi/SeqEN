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
    data_loader = DataLoader()
    SAMPLE_DATA_SIZE = 100
    LARGE_BATCH = 12
    SMALL_BATCH = 4
    DATASET_NAME_seq_ACTp = "Kegg_ndx_ACTp_100"
    DATASET_NAME_seq_ss = "pdb_ndx_ss_100"
    DATASET_NAME_seq_ACTp_ss = "pfam_ndx_wpACT_ss_100"

    def test_load_test_data(self):
        # testing for seq:ACT_p data
        self.data_loader.load_test_data(self.DATASET_NAME_seq_ACTp, self.device)
        assert (
            len(self.data_loader.test_data) == self.SAMPLE_DATA_SIZE
        ), "loaded data size do not match"
        first_item = list(self.data_loader.test_data.values())[0]
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
        self.data_loader.load_test_data(self.DATASET_NAME_seq_ss, self.device)
        assert (
            len(self.data_loader.test_data) == self.SAMPLE_DATA_SIZE
        ), "loaded data size do not match"
        first_item = list(self.data_loader.test_data.values())[0]
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
        # testing for seq:ACT_p:ss data
        # TODO adding tests for trinary dataset seq:ACT_p:ss

    def test_load_train_data(self):
        # training for seq:ACT_p data
        self.data_loader.load_train_data(self.DATASET_NAME_seq_ACTp, self.device)
        assert (
            len(self.data_loader.train_data) == self.SAMPLE_DATA_SIZE
        ), "loaded data size do not match"
        first_item = list(self.data_loader.train_data.values())[0]
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
        # training for seq:ss data
        self.data_loader.load_train_data(self.DATASET_NAME_seq_ss, self.device)
        assert (
            len(self.data_loader.train_data) == self.SAMPLE_DATA_SIZE
        ), "loaded data size do not match"
        first_item = list(self.data_loader.train_data.values())[0]
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
        # testing for seq:ACT_p:ss data
        # TODO adding tests for trinary dataset seq:ACT_p:ss

    def test_get_test_batch(self):
        # testing for seq:ACT_p data
        self.data_loader.load_test_data(self.DATASET_NAME_seq_ACTp, self.device)
        batch_size = 5
        test_batch = [item for item in self.data_loader.get_test_batch(batch_size)]
        assert len(test_batch) == batch_size, "returned wrong number of items"

    def test_get_train_batch(self):
        # training for seq:ACT_p data
        self.data_loader.load_train_data(self.DATASET_NAME_seq_ACTp, self.device)
        batch_num = self.SAMPLE_DATA_SIZE // self.LARGE_BATCH
        large_batch = [
            item for item in self.data_loader.get_train_batch(batch_size=self.LARGE_BATCH)
        ]
        assert len(large_batch) == batch_num, "train batch generator, wrong batch numbers"
        batch_num = self.SAMPLE_DATA_SIZE // self.SMALL_BATCH
        small_batch = [
            item for item in self.data_loader.get_train_batch(batch_size=self.SMALL_BATCH)
        ]
        assert len(small_batch) == batch_num, "train batch generator, wrong batch numbers"
        assert (
            small_batch[0].shape[0] < large_batch[0].shape[0]
        ), "train batch generator, wrong size"
        # max_size
        max_size = int(1.5 * self.SAMPLE_DATA_SIZE)
        train_batch = [
            item
            for item in self.data_loader.get_train_batch(
                batch_size=self.LARGE_BATCH, max_size=max_size
            )
        ]
        assert len(large_batch) < len(train_batch), "problem with max size"
        max_size = int(0.75 * self.SAMPLE_DATA_SIZE)
        train_batch = [
            item
            for item in self.data_loader.get_train_batch(
                batch_size=self.LARGE_BATCH, max_size=max_size
            )
        ]
        assert len(large_batch) > len(train_batch), "problem with max size"


if __name__ == "__main__":
    unittest_main()
