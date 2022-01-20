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

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = device("cuda" if cuda.is_available() else "cpu")
        cls.data_loader = DataLoader()
        cls.SAMPLE_DATA_SIZE = 100
        cls.LARGE_BATCH = 12
        cls.SMALL_BATCH = 4
        cls.DATASET_NAME_seq_ACTp = "kegg_ndx_ACTp_100"
        cls.DATASET_NAME_seq_ss = "pdb_ndx_ss_100"
        cls.DATASET_NAME_seq_ACTp_ss = "pfam_ndx_wpACT_ss_100"

    def test_load_test_data(self):
        # testing for seq:ACT_p data
        self.data_loader.load_test_data(self.DATASET_NAME_seq_ACTp, self.device)
        self.assertEqual(
            len(self.data_loader.test_data), self.SAMPLE_DATA_SIZE, "loaded data size do not match"
        )
        first_item = list(self.data_loader.test_data.values())[0]
        self.assertEqual(len(first_item), 2, "DataLoader must return two items, data and metadata ")
        self.assertIsInstance(
            first_item[0], Tensor, "The first item from DataLoader must be a tensor"
        )
        self.assertEqual(first_item[0].shape[1], 2, "Value's shape from DataLoader must be (-1, 2)")
        self.assertIsInstance(first_item[1], Dict, "The second item from DataLoader must be a dict")
        self.assertEqual(list(first_item[1].keys()), ["name", "0", "1"], "problem with metadata")
        name, ndx, act_p = first_item[1].values()
        self.assertIsInstance(name, str, "name in metadata must be a string")
        self.assertIsInstance(ndx, Dict, "ndx in metadata must be a dict")
        self.assertEqual(ndx["name"], "ndx", "name of second item in metadata must be ndx")
        self.assertIsInstance(act_p, Dict, "act_p in metadata must be a dict")
        self.assertEqual(act_p["name"], "ACT_p", "name of second item in metadata must be ACT_p")
        self.assertEqual(ndx["shape"], act_p["shape"], "ndx and ACT_p shape do not match")
        # testing for seq:ss data
        self.data_loader.load_test_data(self.DATASET_NAME_seq_ss, self.device)
        self.assertEqual(
            len(self.data_loader.test_data), self.SAMPLE_DATA_SIZE, "loaded data size do not match"
        )
        first_item = list(self.data_loader.test_data.values())[0]
        self.assertEqual(
            len(first_item), 2, "DataLoader must return two items, data  and metadata "
        )
        self.assertIsInstance(
            first_item[0], Tensor, "The first item from DataLoader must be a tensor"
        )
        self.assertEqual(
            first_item[0].shape[1], 2, "The value shape from DataLoader must be (-1, 2)"
        )
        self.assertIsInstance(first_item[1], Dict, "The second item from DataLoader must be a dict")
        self.assertEqual(list(first_item[1].keys()), ["name", "0", "1"], "problem with metadata")
        name, ndx, ss = first_item[1].values()
        self.assertIsInstance(name, str, "name in metadata must be a string")
        self.assertIsInstance(ndx, Dict, "ndx in metadata must be a dict")
        self.assertEqual(ndx["name"], "ndx", "name of second item in metadata must be ndx")
        self.assertIsInstance(act_p, Dict, "ss in metadata must be a dict")
        self.assertEqual(ss["name"], "ss", "name of second item in metadata must be ss")
        self.assertEqual(ndx["shape"], ss["shape"], "ndx and ss shape do not match")
        # testing for seq:ACT_p:ss data
        # TODO adding tests for trinary dataset seq:ACT_p:ss

    def test_load_train_data(self):
        # training for seq:ACT_p data
        # testing for seq:ACT_p data
        self.data_loader.load_train_data(self.DATASET_NAME_seq_ACTp, self.device)
        self.assertEqual(
            len(self.data_loader.train_data), self.SAMPLE_DATA_SIZE, "loaded data size do not match"
        )
        first_item = list(self.data_loader.train_data.values())[0]
        self.assertEqual(len(first_item), 2, "DataLoader must return two items, data and metadata ")
        self.assertIsInstance(
            first_item[0], Tensor, "The first item from DataLoader must be a tensor"
        )
        self.assertEqual(first_item[0].shape[1], 2, "Value's shape from DataLoader must be (-1, 2)")
        self.assertIsInstance(first_item[1], Dict, "The second item from DataLoader must be a dict")
        self.assertEqual(list(first_item[1].keys()), ["name", "0", "1"], "problem with metadata")
        name, ndx, act_p = first_item[1].values()
        self.assertIsInstance(name, str, "name in metadata must be a string")
        self.assertIsInstance(ndx, Dict, "ndx in metadata must be a dict")
        self.assertEqual(ndx["name"], "ndx", "name of second item in metadata must be ndx")
        self.assertIsInstance(act_p, Dict, "act_p in metadata must be a dict")
        self.assertEqual(act_p["name"], "ACT_p", "name of second item in metadata must be ACT_p")
        self.assertEqual(ndx["shape"], act_p["shape"], "ndx and ACT_p shape do not match")
        # testing for seq:ss data
        self.data_loader.load_train_data(self.DATASET_NAME_seq_ss, self.device)
        self.assertEqual(
            len(self.data_loader.train_data), self.SAMPLE_DATA_SIZE, "loaded data size do not match"
        )
        first_item = list(self.data_loader.train_data.values())[0]
        self.assertEqual(
            len(first_item), 2, "DataLoader must return two items, data  and metadata "
        )
        self.assertIsInstance(
            first_item[0], Tensor, "The first item from DataLoader must be a tensor"
        )
        self.assertEqual(
            first_item[0].shape[1], 2, "The value shape from DataLoader must be (-1, 2)"
        )
        self.assertIsInstance(first_item[1], Dict, "The second item from DataLoader must be a dict")
        self.assertEqual(list(first_item[1].keys()), ["name", "0", "1"], "problem with metadata")
        name, ndx, ss = first_item[1].values()
        self.assertIsInstance(name, str, "name in metadata must be a string")
        self.assertIsInstance(ndx, Dict, "ndx in metadata must be a dict")
        self.assertEqual(ndx["name"], "ndx", "name of second item in metadata must be ndx")
        self.assertIsInstance(act_p, Dict, "ss in metadata must be a dict")
        self.assertEqual(ss["name"], "ss", "name of second item in metadata must be ss")
        self.assertEqual(ndx["shape"], ss["shape"], "ndx and ss shape do not match")
        # testing for seq:ACT_p:ss data
        # TODO adding tests for trinary dataset seq:ACT_p:ss

    def test_get_test_batch(self):
        # testing for seq:ACT_p data
        self.data_loader.load_test_data(self.DATASET_NAME_seq_ACTp, self.device)
        batch_size = 5
        test_batch = list(self.data_loader.get_test_batch(batch_size))
        self.assertEqual(len(test_batch), batch_size, "returned wrong number of items")

    def test_get_train_batch(self):
        # training for seq:ACT_p data
        self.data_loader.load_train_data(self.DATASET_NAME_seq_ACTp, self.device)
        batch_num = self.SAMPLE_DATA_SIZE // self.LARGE_BATCH
        large_batch = list(self.data_loader.get_train_batch(batch_size=self.LARGE_BATCH))
        self.assertEqual(len(large_batch), batch_num, "train batch generator, wrong batch numbers")
        batch_num = self.SAMPLE_DATA_SIZE // self.SMALL_BATCH
        small_batch = list(self.data_loader.get_train_batch(batch_size=self.SMALL_BATCH))
        self.assertEqual(len(small_batch), batch_num, "train batch generator, wrong batch numbers")
        self.assertLess(
            small_batch[0].shape[0], large_batch[0].shape[0], "train batch generator, wrong size"
        )

    def test_get_train_batch_max_size(self):
        # max_size
        self.data_loader.load_train_data(self.DATASET_NAME_seq_ACTp, self.device)
        large_batch = list(self.data_loader.get_train_batch(batch_size=self.LARGE_BATCH))
        # for max_size bigger than existing data
        max_size = int(1.5 * self.SAMPLE_DATA_SIZE)
        train_batch = list(
            self.data_loader.get_train_batch(batch_size=self.LARGE_BATCH, max_size=max_size)
        )
        self.assertLess(len(large_batch), len(train_batch), "problem with max size")
        # for max_size smaller than existing data
        max_size = int(0.75 * self.SAMPLE_DATA_SIZE)
        train_batch = list(
            self.data_loader.get_train_batch(batch_size=self.LARGE_BATCH, max_size=max_size)
        )
        self.assertGreater(len(large_batch), len(train_batch), "problem with max size")


if __name__ == "__main__":
    unittest_main()
