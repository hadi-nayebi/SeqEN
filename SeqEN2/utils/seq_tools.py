#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from torch import Tensor, diagonal, empty, eye, fliplr, index_select, mode, tensor
from torch.nn.functional import unfold


def consensus(output, ndx, device):
    output_length, w = output.shape
    seq_length = output_length + w - 1
    filter_size = min(seq_length - ndx, ndx + 1)
    if filter_size > w:
        filter_size = w
    r_min = max(0, ndx - w + 1)
    r_max = r_min + filter_size
    r_indices = tensor(range(r_min, r_max), device=device)
    c_min = max(0, ndx - output_length + 1)
    c_max = min(ndx, w - 1) + 1
    c_indices = tensor(range(c_min, c_max), device=device)
    sub_result = index_select(index_select(output, 0, r_indices), 1, c_indices)
    val = mode(diagonal(fliplr(fliplr(eye(filter_size, device=device).long()) * sub_result)))
    return val.values.item()


def get_seq(ndx, ndx_windows):
    output_length, w = ndx_windows.shape
    seq_length = output_length + w - 1
    if ndx < output_length:
        return ndx_windows[ndx][0]
    elif ndx < seq_length:
        return ndx_windows[-1][ndx - output_length + 1]
    else:
        raise IndexError(
            f"index {ndx-output_length+1} is out of bounds for dimension 1 with size {w}"
        )

    # add comment


def get_consensus_seq(seq, device):
    output_length, w = seq.shape
    seq_length = output_length + w - 1
    output = empty(seq_length, device=device)
    for i in range(seq_length):
        output[i] = consensus(seq, i, device=device)
    return output


def consensus_acc(seq, output, device):
    output_length, w = output.shape
    seq_length = output_length + w - 1
    n = 0
    consensus_seq = empty(seq_length, device=device)
    for i in range(seq_length):
        consensus_seq[i] = consensus(output, i, device=device)
        if get_seq(i, seq).item() == consensus_seq[i]:
            n += 1
    return n / seq_length, consensus_seq


def sliding_window(input_vals, w, keys=None):
    assert isinstance(input_vals, Tensor)
    assert input_vals.shape[1] == 1, "input shape must be (-1, 1)"
    kernel_size = (input_vals.shape[1], w)
    input_vals = unfold(input_vals.float().T[None, None, :, :], kernel_size=kernel_size)[0].T
    input_ndx = input_vals[:, :w]
    if keys is not None:
        sliced_seq = []
        for item in input_ndx:
            sliced_seq.append(ndx_to_seq(item, keys))
        return sliced_seq
    return input_ndx


def ndx_to_seq(seq, keys):
    assert isinstance(seq, Tensor)
    return "".join([keys[i] for i in seq.long()])
