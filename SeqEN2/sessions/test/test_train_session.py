#!/usr/bin/env python
# coding: utf-8

"""Unit test TrainSession."""
from pathlib import Path
from unittest import TestCase
from unittest import main as unittest_main

from SeqEN2.sessions.train_session import TrainSession


class TestTrainSession(TestCase):
    """Test items for TrainSession class"""

    train_session = TrainSession()

    def test_load_training_settings(self):
        training_settings_name = "params3"
        training_settings = self.train_session.load_training_settings(training_settings_name)
        self.assertIn("reconstructor", training_settings.keys())


if __name__ == "__main__":
    unittest_main()
