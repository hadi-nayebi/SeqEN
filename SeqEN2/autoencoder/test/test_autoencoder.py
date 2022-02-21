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
from SeqEN2.autoencoder.utils import Architecture
from SeqEN2.model.data_loader import DataLoader, read_json
from SeqEN2.utils.custom_dataclasses import AETrainingSettings, TrainingParams


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
        arch_path = cls.root / "config" / "arch" / "arch5.json"
        arch = Architecture(read_json(str(arch_path)))
        cls.autoencoder = Autoencoder(cls.d1, cls.dn, cls.w, arch)
        cls.data_loader = DataLoader()
        cls.data_loader.load_test_data(cls.DATASET_NAME_seq_ACTp, cls.device)
        cls.data_loader.load_train_data(cls.DATASET_NAME_seq_ACTp, cls.device)
        # random train sample
        cls.train_batch = list(cls.data_loader.get_train_batch(batch_size=10))[0]
        # fixed test sample
        cls.test_batch = cls.data_loader.get_by_key(cls.TEST_KEY)
        cls.autoencoder.to(cls.device)
        # print('from init', cls.autoencoder.training_settings)

    def test_transform_input(self):
        # test batch returns a tuple (data, metadata)
        input_vals = self.test_batch[0]
        input_ndx, one_hot_input = self.autoencoder.transform_input(input_vals, self.device)
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
        # train batch returns data without any metadata
        input_vals = self.train_batch
        input_ndx, one_hot_input = self.autoencoder.transform_input(input_vals, self.device)
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

    def test_forward(self):
        # test batch returns a tuple (data, metadata)
        input_vals = self.test_batch[0]
        input_ndx, one_hot_input = self.autoencoder.transform_input(input_vals, self.device)
        devectorized = self.autoencoder.forward_test(one_hot_input)
        self.assertEqual(
            self.autoencoder.d0, devectorized.shape[1], "output.shape[1] do not match d0"
        )

    def test_train_batch(self):
        # train batch returns data without any metadata
        self.autoencoder.initialize_for_training()
        input_vals = self.train_batch
        self.autoencoder.train_batch(input_vals, self.device)
        # TODO: define a useful assert

    def test_initialize_for_training(self):
        self.autoencoder.training_settings = AETrainingSettings(
            reconstructor=TrainingParams(lr=0.5)
        )
        self.assertEqual(
            self.autoencoder.training_settings.reconstructor.lr, 0.5, "incorrect assignment"
        )
        # passing None do not change anything
        self.autoencoder.initialize_for_training()
        self.assertEqual(
            self.autoencoder.training_settings,
            AETrainingSettings(reconstructor=TrainingParams(lr=0.5)),
            "Incorrect assignment",
        )
        # passing Dict do change anything
        self.autoencoder.initialize_for_training({"reconstructor": TrainingParams(lr=0.7)})
        self.assertEqual(
            self.autoencoder.training_settings,
            AETrainingSettings(reconstructor=TrainingParams(lr=0.7)),
            "Incorrect assignment",
        )

    def test_training_settings(self):
        # type checking, default
        self.assertIsInstance(self.autoencoder.training_settings, AETrainingSettings, "TypeError")

        # assignments
        def assign(ae, value):
            ae.training_settings = value

        self.assertRaises(TypeError, assign, self.autoencoder, "string")
        self.assertRaises(KeyError, assign, self.autoencoder, {"generator": TrainingParams()})
        self.assertRaises(
            KeyError,
            assign,
            self.autoencoder,
            {"reconstructor": TrainingParams(), "generator": TrainingParams()},
        )

    def test_saving_training_settings(self):
        train_dir = self.root / "models" / "dummy"
        self.autoencoder.save_training_settings(train_dir)
        filepath = train_dir / "training_settings.json"
        self.assertEqual((str(filepath), filepath.is_file()), (str(filepath), True))
        system(f"rm {str(filepath)}")

    # def test_update_training_settings(self):
    #     self.autoencoder.training_settings = AETrainingSettings()
    #     print(self.autoencoder.training_settings)
    #     train_dir = self.root / "config" / "train_params"
    #     self.autoencoder.update_training_settings(train_dir)
    #     print(self.autoencoder.training_settings)


if __name__ == "__main__":
    unittest_main()
