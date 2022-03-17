#!/usr/bin/env python
# coding: utf-8

"""Unit test proteins."""
from pathlib import Path
from unittest import TestCase
from unittest import main as unittest_main

from numpy import array

from SeqEN2.protein.protein import Protein


class TestProtein(TestCase):
    """Test items for Proteins class"""

    def test_protein_save_and_rename(self):
        protein = Protein("dummy")
        protein.save_file()
        self.assertTrue(protein.path.exists(), "Saving Failed")
        old_path = Path(str(protein.path))
        protein.rename("new_dummy")
        self.assertFalse(old_path.exists(), "Rename file failed")
        self.assertTrue(protein.path.exists(), "Rename file failed")
        protein.remove_file()
        self.assertFalse(protein.path.exists(), "Remove file failed")

    def test_set_seq(self):
        protein = Protein("dummy")
        # update with correct seq
        protein.seq_ndx = "ACDEHKLLSMSPY"
        seq_ndx = array([7, 12, 14, 13, 17, 19, 5, 5, 10, 3, 10, 9, 1])
        match = all([val1 == val2 for val1, val2 in zip(protein.seq_ndx, seq_ndx)])
        self.assertTrue(match)
        # fail to update with wrong seq
        protein.seq_ndx = "XMJSLLSA"  # must not change the protein.seq_ndx
        match = all([val1 == val2 for val1, val2 in zip(protein.seq_ndx, seq_ndx)])
        self.assertTrue(match)
        # update with correct seq_ndx
        seq_ndx = array([7, 0, 14, 13, 17, 10, 5, 5, 9, 3, 10, 9, 1])
        protein.seq_ndx = seq_ndx
        match = all([val1 == val2 for val1, val2 in zip(protein.seq_ndx, seq_ndx)])
        self.assertTrue(match)
        # fail to update with incorrect seq_ndx
        protein.seq_ndx = array([7, 0, -1, 13, 17, 10, 5, 5, 9, 3, 10, 9, 1])
        match = all([val1 == val2 for val1, val2 in zip(protein.seq_ndx, seq_ndx)])
        self.assertTrue(match)
        # fail to update with incorrect seq_ndx
        protein.seq_ndx = array([7, 0, 4, 13, 17, 10, 5, 5, 22, 3, 10, 9, 1])
        match = all([val1 == val2 for val1, val2 in zip(protein.seq_ndx, seq_ndx)])
        self.assertTrue(match)

    def test_protein_seq_ndx_ss_padding(self):
        protein = Protein("dummy")
        # update with correct seq
        protein.seq_ndx = "*******ACDEHKLLSMSPY*******"
        self.assertEqual(protein.size(), 13)
        self.assertEqual(protein.aa_seq(), "ACDEHKLLSMSPY")
        # ss
        protein.seq_ss = "*******CSTHBEIHBEGHE*******"
        self.assertEqual(protein.ss_seq(), "CSTHBEIHBEGHE")
        # padding
        self.assertEqual(protein.padding, 7)


if __name__ == "__main__":
    unittest_main()
