#!/usr/bin/env python
# coding: utf-8

"""Unit test DataPipeline."""
from unittest import TestCase
from unittest import main as unittest_main

from SeqEN2.model.data_loader import write_json
from SeqEN2.utils.data_pipeline_uniprot import DataPipeline


class TestDataPipeline(TestCase):
    """Test items for DataPipeline class"""

    def test_fetch(self):
        data_pipeline = DataPipeline()
        print("starting fetch")
        seq = """GSPRTVEEVFSDFRGRRAGLIKALSTDVQKFYHQCDPEKENLCLYGLPNETWEVNLPVEEVPPELPEPALGINFARDGMQEKDWISLVAVHSDSWLISVAFYFGARFGFGKNERKRLFQMINDLPTIFEVVTGNA"""
        results = data_pipeline.fetch_by_seq(seq)
        print(results)
        write_json(results, "test.json", pretty=True)


if __name__ == "__main__":
    unittest_main()
