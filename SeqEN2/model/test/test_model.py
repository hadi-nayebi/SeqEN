#!/usr/bin/env python
# coding: utf-8

"""Unit test Model."""

from unittest import TestCase
from unittest import main as unittest_main

from SeqEN2.model.model import Model
from SeqEN2.sessions.train_session import TrainSession


class TestModel(TestCase):
    """Test items for Model class"""

    def test_initialize_training(self):
        train_session = TrainSession()
        train_session.add_model("dummy", "arch7")
        # train_session.model.initialize_training()


if __name__ == "__main__":
    unittest_main()
