#!/usr/bin/env python
# coding: utf-8

"""Unit test DataPipeline."""
from unittest import TestCase
from unittest import main as unittest_main

from SeqEN2.utils.data_pipeline_entrezpy import DataPipeline


class TestDataPipeline(TestCase):
    """Test items for DataPipeline class"""

    def test_fetch(self):
        data_pipeline = DataPipeline("nayebiga@msu.edu")
        print("starting fetch")
        results = data_pipeline.fetch("fumC")

        for i in results:
            print(i, results[i].uid, results[i].caption, results[i].strain, results[i].subtype.host)


if __name__ == "__main__":
    unittest_main()
