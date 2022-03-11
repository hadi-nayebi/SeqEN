#!/usr/bin/env python
# coding: utf-8

# by nayebiga@msu.edu
__version__ = "0.0.1"

from numpy.random import choice
from torch import (
    Tensor,
    argmax,
    cat,
    diagonal,
    empty,
    eye,
    fliplr,
    index_select,
    mode,
    randperm,
)
from torch import sum as torch_sum
from torch import tensor
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
    consensus_seq = empty(seq_length, device=device)
    for i in range(seq_length):
        consensus_seq[i] = consensus(seq, i, device=device)
    return consensus_seq


def consensus_acc(seq, output, w, device):
    output = output_to_ndx(output, w)
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


def output_to_ndx(output, w):
    return argmax(output, dim=1).reshape((-1, w))


def reconstructor_acc(output, input_ndx):
    return torch_sum(argmax(output, dim=1) == input_ndx.reshape((-1,))) / output.shape[0]


def add_noise(one_hot_input, input_noise, device):
    ndx = randperm(one_hot_input.shape[1])
    size = list(one_hot_input.shape)
    size[-1] = 1
    p = tensor(choice([1, 0], p=[input_noise, 1 - input_noise], size=size)).to(device)
    return (one_hot_input[:, ndx, :] * p) + (one_hot_input * (1 - p))


def slide_window(input_vals, w):
    kernel_size = (input_vals.shape[1], w)
    return unfold(input_vals.float().T[None, None, :, :], kernel_size=kernel_size)[0].T


def split_input_vals(input_vals, input_keys, w):
    target_vals_ss = None
    target_vals_cl = None
    if len(input_keys) == 2:
        if input_keys[1] == "S":
            target_vals_ss = input_vals[:, w:].long()
        elif input_keys[1] == "C":
            target_vals_cl = input_vals[:, w:].mean(axis=1).reshape((-1, 1))
            target_vals_cl = cat((target_vals_cl, 1 - target_vals_cl), 1).float()
    elif len(input_keys) == 3:
        if input_keys[1] == "S":
            target_vals_ss = input_vals[:, w:-w].long()
        elif input_keys[2] == "S":
            target_vals_ss = input_vals[:, -w:].long()
        if input_keys[1] == "C":
            target_vals_cl = input_vals[:, w:-w].mean(axis=1).reshape((-1, 1))
            target_vals_cl = cat((target_vals_cl, 1 - target_vals_cl), 1).float()
        elif input_keys[2] == "C":
            target_vals_cl = input_vals[:, -w:].mean(axis=1).reshape((-1, 1))
            target_vals_cl = cat((target_vals_cl, 1 - target_vals_cl), 1).float()
    return target_vals_ss, target_vals_cl


def continuity_target_right(encoded_output):
    return cat((encoded_output[1:], encoded_output[-1].unsqueeze(0)), 0)


def continuity_target_left(encoded_output):
    return cat((encoded_output[0].unsqueeze(0), encoded_output[:-1]), 0)
