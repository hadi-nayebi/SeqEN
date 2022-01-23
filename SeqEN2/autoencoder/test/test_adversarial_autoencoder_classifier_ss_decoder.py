#!/usr/bin/env python
# coding: utf-8

"""Unit test AAECSS."""

from os.path import dirname
from pathlib import Path
from unittest import TestCase
from unittest import main as unittest_main

from torch import cuda, device

from SeqEN2.autoencoder.adversarial_autoencoder_classifier_ss_decoder import (
    AdversarialAutoencoderClassifierSSDecoder,
)
from SeqEN2.autoencoder.utils import Architecture
from SeqEN2.model.data_loader import DataLoader, read_json
from SeqEN2.utils.custom_dataclasses import AAECSSTrainingSettings, TrainingParams


class TestAutoencoder(TestCase):
    """Test items for AAECSS class"""

    root = Path(dirname(__file__)).parent.parent.parent
    device = device("cuda" if cuda.is_available() else "cpu")
    DATASET_NAME_seq_ACTp = "kegg_ndx_ACTp_100"
    DATASET_NAME_seq_ss = "pdb_ndx_ss_100"
    autoencoder = None
    data_loader_ss = None
    data_loader_cl = None
    w = 20
    dn = 10
    d1 = 8
    TEST_KEY_CL = "CO657_07215"
    TEST_KEY_SS = "232L_1_A"

    @classmethod
    def setUpClass(cls) -> None:
        # replace arch1.json to test other ones
        arch_path = cls.root / "config" / "arch" / "arch6.json"
        arch = Architecture(read_json(str(arch_path)))
        cls.autoencoder = AdversarialAutoencoderClassifierSSDecoder(cls.d1, cls.dn, cls.w, arch)
        cls.data_loader_cl = DataLoader()
        cls.data_loader_ss = DataLoader()
        cls.data_loader_cl.load_test_data(cls.DATASET_NAME_seq_ACTp, cls.device)
        cls.data_loader_cl.load_train_data(cls.DATASET_NAME_seq_ACTp, cls.device)
        cls.data_loader_ss.load_test_data(cls.DATASET_NAME_seq_ss, cls.device)
        cls.data_loader_ss.load_train_data(cls.DATASET_NAME_seq_ss, cls.device)
        # random train sample
        cls.train_batch_cl = list(cls.data_loader_cl.get_train_batch(batch_size=10))[0]
        cls.train_batch_ss = list(cls.data_loader_ss.get_train_batch(batch_size=10))[0]
        # fixed test sample
        cls.test_batch_cl = cls.data_loader_cl.get_test_by_key(cls.TEST_KEY_CL)
        cls.test_batch_ss = cls.data_loader_ss.get_test_by_key(cls.TEST_KEY_SS)
        cls.autoencoder.to(cls.device)

    def test_transform_input(self):
        # test batch returns a tuple (data, metadata)
        input_vals = self.test_batch_ss[0]
        input_ndx, target_vals, one_hot_input = self.autoencoder.transform_input_ss(
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
        self.assertEqual(self.w, target_vals.shape[1], "target_vals.shape[0] do not match w")
        # train batch returns data without any metadata
        input_vals = self.train_batch_ss
        input_ndx, target_vals, one_hot_input = self.autoencoder.transform_input_ss(
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
        self.assertEqual(self.w, target_vals.shape[1], "target_vals.shape[0] do not match w")

    def test_forward(self):
        # test batch returns a tuple (data, metadata)
        input_vals = self.test_batch_ss[0]
        input_ndx, target_vals, one_hot_input = self.autoencoder.transform_input_ss(
            input_vals, self.device
        )
        (
            devectorized,
            discriminator_output,
            classifier_output,
            ss_decoder_output,
        ) = self.autoencoder.forward_test(one_hot_input)
        self.assertEqual(
            self.autoencoder.d0, devectorized.shape[1], "output.shape[1] do not match d0"
        )
        self.assertEqual(2, discriminator_output.shape[1], "output2.shape[1] do not match two")
        self.assertEqual(2, classifier_output.shape[1], "output3.shape[1] do not match two")
        self.assertEqual(
            self.autoencoder.ds, ss_decoder_output.shape[1], "output4.shape[1] do not match ds"
        )

    def test_train_batch(self):
        # train batch returns data without any metadata
        self.autoencoder.initialize_for_training()
        input_vals = {"cl": self.train_batch_cl, "ss": self.train_batch_ss}
        self.autoencoder.train_batch(input_vals, self.device)

        # TODO: define a useful assert

    def test_initialize_for_training(self):
        self.autoencoder.training_settings = AAECSSTrainingSettings(
            classifier=TrainingParams(lr=0.5)
        )
        self.assertEqual(
            self.autoencoder.training_settings.classifier.lr, 0.5, "incorrect assignment"
        )
        # passing None do not change anything
        self.autoencoder.initialize_for_training()
        self.assertEqual(
            self.autoencoder.training_settings,
            AAECSSTrainingSettings(classifier=TrainingParams(lr=0.5)),
            "Incorrect assignment",
        )
        # passing Dict do change anything
        self.autoencoder.initialize_for_training({"classifier": TrainingParams(lr=0.7)})
        self.assertEqual(
            self.autoencoder.training_settings,
            AAECSSTrainingSettings(classifier=TrainingParams(lr=0.7)),
            "Incorrect assignment",
        )

    def test_training_settings(self):
        # type checking, default
        self.assertIsInstance(
            self.autoencoder.training_settings, AAECSSTrainingSettings, "TypeError"
        )

        # assignments
        def assign(ae, value):
            ae.training_settings = value

        self.assertRaises(TypeError, assign, self.autoencoder, 3.14)
        self.assertRaises(KeyError, assign, self.autoencoder, {"unknown": TrainingParams()})
        self.assertRaises(
            KeyError,
            assign,
            self.autoencoder,
            {"reconstructor": TrainingParams(lr=1.0), "unknown": TrainingParams()},
        )


if __name__ == "__main__":
    unittest_main()
