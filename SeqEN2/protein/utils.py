import pickle
from collections.abc import Iterable

from numpy import array, int8
from torch import Tensor, tensor

KEYS = {"aa_keys": "WYFMILVAGPSTCEDQNHRK*", "ss_keys": "CSTIGHBE*"}
AA_PADDING_NDX = 20
SS_PADDING_NDX = 8


def is_ss_sequence(seq):
    if isinstance(seq, str):
        return all([char in KEYS["ss_keys"] for char in seq])
    return False


def is_protein_sequence(seq):
    if isinstance(seq, str):
        return all([char in KEYS["aa_keys"] for char in seq])
    return False


def is_array_int(seq, max_val=20):
    if isinstance(seq, Iterable) and not isinstance(seq, str):
        seq = array(seq, dtype=int8)
        is_min_zero = min(seq) >= 0
        is_max_met = max(seq) <= max_val
        are_all_int = all([isinstance(num, int8) for num in seq])
        return is_min_zero and is_max_met and are_all_int
    return False


def is_array_p(value):
    if isinstance(value, Iterable):
        value = array(value, dtype=float)
        is_min_zero = min(value) >= 0.0
        is_max_one = max(value) <= 1.0
        are_all_float = all([isinstance(num, float) for num in value])
        return is_min_zero and is_max_one and are_all_float
    return False


def seq_to_ndx(seq, keys="aa_keys"):
    return array([KEYS[keys].index(c) for c in seq], dtype=int8)


def ndx_to_seq(seq, keys="aa_keys"):
    # TODO use numpy array instead of torch tensor
    if not isinstance(seq, Tensor):
        seq = tensor(seq)
    return "".join([KEYS[keys][i] for i in seq.long()])


def load_protein(path):
    with open(path, "rb") as f:
        return pickle.load(f)
