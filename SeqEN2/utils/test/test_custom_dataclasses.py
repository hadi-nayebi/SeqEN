#!/usr/bin/env python
# coding: utf-8

"""Unit test Validations."""

from unittest import TestCase
from unittest import main as unittest_main

from SeqEN2.utils.custom_dataclasses import *


class TestCustomDataclasses(TestCase):
    """Test items for Validations class"""

    def test_training_params(self):
        # defaults
        tp = TrainingParams()
        self.assertEqual(0.01, tp.lr, "The default value for learning rate is wrong.")
        self.assertEqual(0.9, tp.factor, "The default value for factor is wrong.")
        self.assertEqual(10000, tp.patience, "The default value for patience is wrong.")
        self.assertEqual(0.00001, tp.min_lr, "The default value for min lr is wrong.")
        # assignments
        tp = TrainingParams(lr=0.1, factor=0.8, patience=100, min_lr=0.01)
        self.assertEqual(0.1, tp.lr, "Value assignment for learning rate is wrong.")
        self.assertEqual(0.8, tp.factor, "Value assignment for factor is wrong.")
        self.assertEqual(100, tp.patience, "Value assignment for patience is wrong.")
        self.assertEqual(0.01, tp.min_lr, "Value assignment for min lr is wrong.")

    def test_ae_training_settings(self):
        # defaults
        ae_ts = AETrainingSettings()
        default_tp = TrainingParams()
        self.assertEqual(
            ae_ts.reconstructor, default_tp, "The default value for reconstructor is wrong."
        )
        # assignments
        ae_ts = AETrainingSettings(reconstructor=TrainingParams(lr=0.1))
        self.assertEqual(
            0.1, ae_ts.reconstructor.lr, "Value assignment for learning rate is wrong."
        )
        self.assertEqual(0.9, ae_ts.reconstructor.factor, "Value assignment for factor is wrong.")

    def test_aae_training_settings(self):
        pass


if __name__ == "__main__":
    unittest_main()
