#!/usr/bin/env python
# coding: utf-8

"""Unit test AE."""

from os.path import dirname
from pathlib import Path
from unittest import TestCase
from unittest import main as unittest_main

from torch import cuda, device

from SeqEN2.autoencoder.adversarial_autoencoder_classifier import (
    AdversarialAutoencoderClassifier,
)
from SeqEN2.autoencoder.utils import Architecture
from SeqEN2.model.data_loader import DataLoader, read_json


class TestAutoencoder(TestCase):
    """Test items for AE class"""

    root = Path(dirname(__file__)).parent.parent.parent
    device = device("cuda" if cuda.is_available() else "cpu")
    DATASET_NAME_seq_ACTp = "kegg_ndx_ACTp_100"
    autoencoder = None
    data_loader = None
    w = 20
    dn = 10
    d1 = 8
    TEST_KEY = "CO657_07215"

    @classmethod
    def setUpClass(cls) -> None:
        # replace arch1.json to test other ones
        arch_path = cls.root / "config" / "arch" / "arch3.json"
        arch = Architecture(read_json(str(arch_path)))
        cls.autoencoder = AdversarialAutoencoderClassifier(cls.d1, cls.dn, cls.w, arch)
        cls.data_loader = DataLoader()
        cls.data_loader.load_test_data(cls.DATASET_NAME_seq_ACTp, cls.device)
        cls.data_loader.load_train_data(cls.DATASET_NAME_seq_ACTp, cls.device)
        # random train sample
        cls.train_batch = list(cls.data_loader.get_train_batch(batch_size=10))[0]
        # fixed test sample
        cls.test_batch = cls.data_loader.get_test_by_key(cls.TEST_KEY)

    def test_transform_input(self):
        # test batch returns a tuple (data, metadata)
        input_vals = self.test_batch[0]
        input_ndx, target_vals, one_hot_input = self.autoencoder.transform_input(
            input_vals, self.device
        )
        self.assertEqual(
            input_vals.shape[0] - self.w + 1,
            input_ndx.shape[0],
            "input_ndx.shape[0] do not match batch.shape[0]",
        )
        self.assertEqual(self.w, input_ndx.shape[1], "input_ndx.shape[0] do not match w")
        self.assertEqual(self.w, one_hot_input.shape[1], "one_hot.shape[0] do not match w")
        self.assertEqual(
            self.autoencoder.d0, one_hot_input.shape[2], "one_hot.shape[1] do not match d0"
        )
        self.assertEqual(
            input_vals.shape[0] - self.w + 1,
            target_vals.shape[0],
            "target_vals.shape[0] do not match batch.shape[0]",
        )
        self.assertEqual(2, target_vals.shape[1], "target_vals.shape[0] do not match two")
        # train batch returns data without any metadata
        input_vals = self.train_batch
        input_ndx, target_vals, one_hot_input = self.autoencoder.transform_input(
            input_vals, self.device
        )
        self.assertEqual(
            input_vals.shape[0] - self.w + 1,
            input_ndx.shape[0],
            "input_ndx.shape[0] do not match batch.shape[0]",
        )
        self.assertEqual(self.w, input_ndx.shape[1], "input_ndx.shape[0] do not match w")
        self.assertEqual(self.w, one_hot_input.shape[1], "one_hot.shape[0] do not match w")
        self.assertEqual(
            self.autoencoder.d0, one_hot_input.shape[2], "one_hot.shape[1] do not match d0"
        )
        self.assertEqual(
            input_vals.shape[0] - self.w + 1,
            target_vals.shape[0],
            "target_vals.shape[0] do not match batch.shape[0]",
        )
        self.assertEqual(2, target_vals.shape[1], "target_vals.shape[0] do not match two")

    def test_forward(self):
        # test batch returns a tuple (data, metadata)
        input_vals = self.test_batch[0]
        input_ndx, target_vals, one_hot_input = self.autoencoder.transform_input(
            input_vals, self.device
        )
        devectorized, discriminator_output, classifier_output = self.autoencoder.forward_test(
            one_hot_input
        )
        self.assertEqual(
            self.autoencoder.d0, devectorized.shape[1], "output.shape[1] do not match d0"
        )
        self.assertEqual(2, discriminator_output.shape[1], "output2.shape[1] do not match two")
        self.assertEqual(2, classifier_output.shape[1], "output3.shape[1] do not match two")
        # train batch returns data without any metadata
        input_vals = self.train_batch
        input_ndx, target_vals, one_hot_input = self.autoencoder.transform_input(
            input_vals, self.device
        )
        devectorized, discriminator_output, classifier_output = self.autoencoder.forward_test(
            one_hot_input
        )
        self.assertEqual(
            self.autoencoder.d0, devectorized.shape[1], "output.shape[1] do not match d0"
        )
        self.assertEqual(2, discriminator_output.shape[1], "output2.shape[1] do not match two")
        self.assertEqual(2, classifier_output.shape[1], "output3.shape[1] do not match two")


if __name__ == "__main__":
    unittest_main()
