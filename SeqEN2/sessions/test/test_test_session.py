#!/usr/bin/env python
# coding: utf-8

"""Unit test TestSession."""
from typing import Dict
from unittest import TestCase
from unittest import main as unittest_main

from pandas import DataFrame
from torch.nn import Sequential

from SeqEN2.sessions.test_session import TestSession


class TestTestSession(TestCase):
    """Test items for TestSession class"""

    test_session = TestSession()
    DATASET_NAME = "pdb_act_clss_train"
    MODEL_NAME = "experiment0125"
    ARCH = "arch7"
    VERSION = "202201261847_AAECSS_arch7"
    MODEL_ID = 300
    TEST_ITEMS = 10

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_session.add_model(
            cls.MODEL_NAME, cls.ARCH, cls.VERSION, cls.MODEL_ID, d1=8, dn=10, w=20
        )
        cls.test_session.load_data("clss", cls.DATASET_NAME)
        cls.test_session.get_embedding(num_test_items=cls.TEST_ITEMS)

    def test_add_model(self):
        self.assertIsInstance(self.test_session.model.autoencoder.vectorizer, Sequential)
        self.assertIsInstance(self.test_session.model.autoencoder.devectorizer, Sequential)
        self.assertIsInstance(self.test_session.model.autoencoder.encoder, Sequential)
        self.assertIsInstance(self.test_session.model.autoencoder.decoder, Sequential)
        # self.assertIsInstance(self.test_session.model.autoencoder.ss_decoder, Sequential)
        # self.assertIsInstance(self.test_session.model.autoencoder.classifier, Sequential)
        # self.assertIsInstance(self.test_session.model.autoencoder.discriminator, Sequential)

    def test_get_embedding(self):
        if self.TEST_ITEMS != -1:
            self.assertEqual(self.TEST_ITEMS, len(self.test_session.embedding_results))
        one_embedding = list(self.test_session.embedding_results.values())[0]
        self.assertIsInstance(self.test_session.embedding_results, Dict)
        self.assertIsInstance(one_embedding, DataFrame)
        # print(one_embedding.attrs)
        # print(one_embedding.columns)
        # print(one_embedding)

    def test_tsne_embeddings(self):
        # pass
        self.test_session.plot_embedding_2d(auto_open=True)
        # opens two html files


if __name__ == "__main__":
    unittest_main()
