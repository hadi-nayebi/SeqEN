#!/usr/bin/env python
# coding: utf-8

"""Unit test Model."""

from unittest import TestCase
from unittest import main as unittest_main

from SeqEN2.sessions.train_session import TrainSession


class TestAutoencoder(TestCase):
    """Test items for Autoencoder class"""

    initialized = False

    def initialize(self):
        if not self.initialized:
            self.train_session = TrainSession()
            self.train_session.add_model("dummy", "gen3", "AE", d0=21, d1=8, dn=10, w=20)
            dataset_name = "KeggSeq_ndx_wpACT_100"
            self.train_session.load_data(dataset_name)
            self.initialized = True

    def test_transform_input(self):
        self.initialize()
        w = self.train_session.model.w
        d0 = self.train_session.model.d0
        test_batch = [_ for _ in self.train_session.model.data_loader.get_test_batch(batch_size=1)][
            0
        ]
        (
            input_ndx,
            target_vals,
            one_hot_input,
        ) = self.train_session.model.autoencoder.transform_input(
            test_batch, self.train_session.model.device
        )
        assert test_batch.shape[0] - w + 1 == input_ndx.shape[0]
        assert w == input_ndx.shape[1]
        assert target_vals.shape[0] == input_ndx.shape[0]
        assert target_vals.shape[1] == 1
        assert target_vals.shape[0] == one_hot_input.shape[0]
        assert w == one_hot_input.shape[1]
        assert d0 == one_hot_input.shape[2]


if __name__ == "__main__":
    unittest_main()
