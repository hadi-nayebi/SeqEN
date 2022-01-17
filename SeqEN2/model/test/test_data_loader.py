#!/usr/bin/env python
# coding: utf-8

"""Unit test DataLoader."""

from unittest import TestCase
from unittest import main as unittest_main

from numpy import ndarray

from SeqEN2.model.data_loader import DataLoader


class TestDataLoader(TestCase):
    """Test items for DataLoader class"""

    def test_load_test_data(self):
        dataset_name = "KeggSeq_ndx_wpACT_100"
        data_loader = DataLoader()
        data_loader.load_test_data(dataset_name)
        assert isinstance([item for item in data_loader.test_data.values()][0], ndarray)

    def test_load_train_data(self):
        dataset_name = "KeggSeq_ndx_wpACT_100"
        data_loader = DataLoader()
        data_loader.load_train_data(dataset_name)
        assert isinstance([item for item in data_loader.train_data.values()][0], ndarray)

    def test_get_test_batch(self):
        dataset_name = "KeggSeq_ndx_wpACT_100"
        data_loader = DataLoader()
        data_loader.load_test_data(dataset_name)
        batch_size = 5
        assert (
            len([item for item in data_loader.get_test_batch(batch_size)]) == batch_size
        ), "returned wrong  number of items"

    def test_get_train_batch(self):
        dataset_name = "KeggSeq_ndx_wpACT_100"
        data_loader = DataLoader()
        data_loader.load_train_data(dataset_name)
        large_batch = small_batch = 0
        for item in data_loader.get_train_batch(10):
            large_batch = item.shape
            break
        for item in data_loader.get_train_batch(1):
            small_batch = item.shape
            break
        assert large_batch > small_batch, "train batch fails"


if __name__ == "__main__":
    unittest_main()
