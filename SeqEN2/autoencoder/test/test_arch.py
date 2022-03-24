#!/usr/bin/env python
# coding: utf-8

"""Unit test AE."""
from os import system
from os.path import dirname
from pathlib import Path
from unittest import TestCase
from unittest import main as unittest_main

from torch import cuda, device

from SeqEN2.autoencoder.autoencoder import Autoencoder
from SeqEN2.autoencoder.autoencoder_classifier_ss_decoder import (
    AutoencoderClassifierSSDecoder,
)
from SeqEN2.autoencoder.utils import Architecture
from SeqEN2.model.data_loader import DataLoader, read_json


class TestAutoencoderArch(TestCase):
    """Test items for Arch class"""

    root = Path(dirname(__file__)).parent.parent.parent
    device = device("cuda" if cuda.is_available() else "cpu")
    DATASET_NAME_clss = "pdb_act_clss"
    autoencoder = None
    data_loader = None
    w = 10
    dn = 3
    d1 = 8
    TEST_KEY = "6U9H"
    ARCH = "arch100.json"  # for FCN
    # ARCH = "arch36.json"  # for Conv

    @classmethod
    def setUpClass(cls) -> None:
        # replace arch1.json to test other ones
        arch_path = cls.root / "config" / "arch" / cls.ARCH
        arch = Architecture(read_json(str(arch_path)))
        cls.autoencoder = AutoencoderClassifierSSDecoder(cls.d1, cls.dn, cls.w, arch)
        cls.data_loader = DataLoader()
        cls.data_loader.load_test_data(cls.DATASET_NAME_clss, cls.device)
        cls.data_loader.load_train_data(cls.DATASET_NAME_clss, cls.device)
        # random train sample
        cls.train_batch = list(cls.data_loader.get_train_batch(batch_size=1))
        # fixed test sample
        cls.test_batch = cls.data_loader.get_by_key(cls.TEST_KEY)
        cls.autoencoder.to(cls.device)
        # print('from init', cls.autoencoder.training_settings)

    def test_forward(self):
        # test batch returns a tuple (data, metadata)
        input_vals = self.train_batch[10]
        input_ndx, _, _, one_hot_input = self.autoencoder.transform_input(input_vals, self.device)
        (
            devectorized,
            classifier_output,
            ss_decoder_output,
            encoded,
        ) = self.autoencoder.unit_test_forward(one_hot_input)
        self.assertEqual(
            self.autoencoder.d0, devectorized.shape[1], "output.shape[1] do not match d0"
        )
        self.assertEqual(self.autoencoder.dn, encoded.shape[1], "encoded.shape[1] do not match dn")
        self.assertEqual(
            self.autoencoder.ds,
            ss_decoder_output.shape[1],
            "ss_decoder_output.shape[1] do not match ds",
        )
        self.assertEqual(2, classifier_output.shape[1], "classifier_output.shape[1] do not match 2")


if __name__ == "__main__":
    unittest_main()
