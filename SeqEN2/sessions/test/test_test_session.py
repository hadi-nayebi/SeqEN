#!/usr/bin/env python
# coding: utf-8

"""Unit test TestSession."""
from typing import Dict
from unittest import TestCase
from unittest import main as unittest_main

from pandas import DataFrame
from torch.nn import Sequential

from SeqEN2.sessions.test_session import TestSession


class TestTrainSession(TestCase):
    """Test items for TestSession class"""

    test_session = TestSession()
    DATASET_NAME_seq_ACTp = "kegg_ndx_ACTp_100"
    DATASET_NAME_seq_ss = "pdb_ndx_ss_100"
    MODEL_NAME = "dummy"
    ARCH = "arch6"
    VERSION = "test_AAECSS_arch6"
    MODEL_ID = 0

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_session.add_model(
            cls.MODEL_NAME, cls.ARCH, cls.VERSION, cls.MODEL_ID, d1=8, dn=10, w=20
        )
        cls.test_session.load_data("cl", cls.DATASET_NAME_seq_ACTp)
        cls.test_session.get_embedding(num_test_items=10)

    def test_add_model(self):
        self.assertIsInstance(self.test_session.model.autoencoder.vectorizer, Sequential)
        self.assertIsInstance(self.test_session.model.autoencoder.encoder, Sequential)
        self.assertIsInstance(self.test_session.model.autoencoder.decoder, Sequential)
        self.assertIsInstance(self.test_session.model.autoencoder.ss_decoder, Sequential)
        self.assertIsInstance(self.test_session.model.autoencoder.classifier, Sequential)
        self.assertIsInstance(self.test_session.model.autoencoder.discriminator, Sequential)

    def test_get_embedding(self):
        self.assertEqual(10, len(self.test_session.embedding_results))
        one_embedding = list(self.test_session.embedding_results.values())[0]
        self.assertIsInstance(self.test_session.embedding_results, Dict)
        self.assertIsInstance(one_embedding, DataFrame)
        self.assertIn("act_pred", one_embedding.columns, "missing a column un results")
        self.assertIn("act_trg", one_embedding.columns, "missing a column un results")
        self.assertIn("slices", one_embedding.columns, "missing a column un results")
        self.assertIn("embedding", one_embedding.columns, "missing a column un results")

    def test_tsne_embeddings(self):
        pass
        # self.test_session.plot_embedding_2d()
        # opens two html files


if __name__ == "__main__":
    unittest_main()
