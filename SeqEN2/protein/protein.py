import pickle
from collections.abc import Iterable
from os import system
from os.path import dirname
from pathlib import Path

from numpy import array, int8

from SeqEN2.protein.exceptions import ImmutablePropertyError
from SeqEN2.protein.utils import (
    AA_PADDING_NDX,
    SS_PADDING_NDX,
    is_array_int,
    is_array_p,
    is_protein_sequence,
    is_ss_sequence,
    ndx_to_seq,
    seq_to_ndx,
)
from SeqEN2.utils.base_logger import base_logger
from SeqEN2.utils.data_pipeline_uniprot import DataPipeline


class Protein:
    """template for proteins to package the data used in SeqEn"""

    root = Path(dirname(__file__)).parent.parent

    def __init__(self, name):
        self._name = name
        # define path to save the pickle
        self._path = self.root / "proteins" / f"{self.name}.pkl.bz2"
        # Protein attrs
        self._seq_ndx = None  # seq index used for training
        self._padding = None
        self._checksum = None
        self._seq_ss = None  # ss index used for training
        self._annotations = {}  # annotations used for training
        self._dataset = {"name": "", "mode": ""}  # dataset attrs used for the training purposes
        self.metadata = {}
        # add name to metadata
        self.add_metadata("names", {name})

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path

    @property
    def seq_ndx(self):
        return self._seq_ndx

    @property
    def checksum(self):
        return self._checksum

    @property
    def seq_ss(self):
        return self._seq_ss

    @property
    def annotations(self):
        return self._annotations

    @property
    def dataset(self):
        return self._dataset

    @property
    def padding(self):
        return self._padding

    @seq_ndx.setter
    def seq_ndx(self, seq):
        try:
            if self._checksum is not None:
                raise ImmutablePropertyError(attr="Protein sequence")
            if is_protein_sequence(seq):
                self._seq_ndx = seq_to_ndx(seq, keys="aa_keys")
            elif is_array_int(seq, max_val=AA_PADDING_NDX):
                self._seq_ndx = array(seq, dtype=int8)
            self._padding = len(self._seq_ndx[self._seq_ndx == AA_PADDING_NDX]) // 2
        except ImmutablePropertyError:
            base_logger.logger.info(f"Request to update {self.name} sequence failed.")

    @checksum.setter
    def checksum(self, value):
        try:
            if self._checksum is not None:
                if value == self._checksum:
                    return
                raise ImmutablePropertyError(attr="Checksum")
            self._checksum = value
        except ImmutablePropertyError:
            base_logger.logger.info(f"Request to update {self.name} checksum failed.")

    @seq_ss.setter
    def seq_ss(self, seq):
        # TODO cases were ss_seq is entered with no padding?
        if is_ss_sequence(seq):
            assert len(seq) == len(self._seq_ndx)
            self._seq_ss = seq_to_ndx(seq, keys="ss_keys")
        elif is_array_int(seq, max_val=SS_PADDING_NDX):
            assert len(seq) == len(self._seq_ndx)
            self._seq_ss = array(seq, dtype=int8)

    @annotations.setter
    def annotations(self, value):
        # TODO cases were annotations are entered with no padding?
        if isinstance(value, dict):
            for key, item in value.items():
                if is_array_p(item):
                    assert len(item) == len(self._seq_ndx)
                    self._annotations[key] = array(item, dtype=float)

    @dataset.setter
    def dataset(self, value):
        if isinstance(value, dict):
            assert "name" in value.keys()
            assert "mode" in value.keys()
            self._dataset["name"] = value["name"]
            self._dataset["mode"] = value["mode"]

    def rename(self, new_name, save=False):
        if new_name != self.name:
            if save:
                self.remove_file()
            self._name = new_name
            self._path = self.root / "proteins" / f"{self.name}.pkl.bz2"
            self.add_metadata("names", {new_name})
            if save:
                self.save_file(overwrite=True)

    def remove_file(self):
        if self._path.exists():
            system(f"rm {self._path}")

    def save_file(self, overwrite=False):
        if not overwrite:
            i = 0
            original_name = self._name[:]
            while self._path.exists():
                i += 1
                self.rename(f"{original_name}_{i}")
        with open(self._path, "wb") as f:
            pickle.dump(self, f)

    def add_metadata(self, key, value):
        if key in self.metadata.keys():
            if isinstance(self.metadata[key], list):
                if isinstance(value, list):
                    for item in value:
                        self.metadata[key].append(item)
                else:
                    self.metadata[key].append(value)
            if isinstance(self.metadata[key], set):
                if isinstance(value, set):
                    for item in value:
                        self.metadata[key].add(item)
                else:
                    self.metadata[key].add(value)
        else:
            self.metadata[key] = value

    def del_metadata(self, key):
        del self.metadata[key]

    def aa_seq(self, no_padding=True):
        if no_padding:
            return ndx_to_seq(self._seq_ndx, keys="aa_keys").replace("*", "")
        return ndx_to_seq(self._seq_ndx)

    def ss_seq(self, no_padding=True):
        if no_padding:
            return ndx_to_seq(self._seq_ss, keys="ss_keys").replace("*", "")
        return ndx_to_seq(self._seq_ss)

    def size(self, padding_ndx=20):
        return len(self._seq_ndx[self._seq_ndx != padding_ndx])

    def __str__(self):
        output = f"name: {self.name}\n"
        output += f"seq: {self.aa_seq()}\n"
        output += f"checksum: {self._checksum}\n"
        output += f"dataset: {self.dataset}\n"
        has_ss = "has_ss" if self._seq_ss is not None else "no_ss"
        has_padding = f"padded by {self._padding}" if self._padding > 0 else ""
        output += f"attrs: {has_ss} | {has_padding}\n"
        output += f"annotations: {', '.join(self._annotations.keys()) if len(self._annotations) > 0 else ''}\n"
        output += (
            f"metadata keys: {', '.join(self.metadata.keys()) if len(self.metadata) > 0 else ''}\n"
        )
        return output

    def get_data(self):
        data = {"ndx": self._seq_ndx, "ss": self._seq_ss}
        for key, value in self._annotations.items():
            data[key] = value
        return data

    def update_uniprot_metadata(self, db_ref_limit=20):
        data_pipeline = DataPipeline()
        aa_seq = self.aa_seq()
        try:
            result = data_pipeline.fetch_by_seq(aa_seq)
            # parse and add uniprot metadata to metadata
            if aa_seq != result["sequence"]["content"]:
                base_logger.logger.info(f"sequence for {self.name} does not match uniprot results")
                return
            self.checksum = result["sequence"]["checksum"]
            # update name
            self.rename(result["accession"])
            # store signatureSequenceMatch
            self.add_metadata("UP_signatureSequenceMatch", result["signatureSequenceMatch"])
            # store dbReference
            state = "complete" if len(result["dbReference"]) < db_ref_limit else "partial"
            up_db_reference = {"value": result["dbReference"][:db_ref_limit], "state": state}
            self.add_metadata("UP_dbReference", up_db_reference)
        except KeyError:
            base_logger.logger.info(f"{self.name} failed to get uniprot metadata.")
