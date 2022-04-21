"""
Losses for training neural networks.

Authors
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Hwidong Na 2020
 * Yan Gao 2020
 * Titouan Parcollet 2020
"""

import math
import torch
import logging
import functools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.decoders.ctc import filter_ctc_output
import librosa


class PitWrapper(nn.Module):
    """
    Permutation Invariant Wrapper to allow Permutation Invariant Training
    (PIT) with existing losses.

    Permutation invariance is calculated over the sources/classes axis which is
    assumed to be the rightmost dimension: predictions and targets tensors are
    assumed to have shape [batch, ..., channels, sources].

    Arguments
    ---------
    base_loss : function
        Base loss function, e.g. torch.nn.MSELoss. It is assumed that it takes
        two arguments:
        predictions and targets and no reduction is performed.
        (if a pytorch loss is used, the user must specify reduction="none").

    Returns
    ---------
    pit_loss : torch.nn.Module
        Torch module supporting forward method for PIT.

    Example
    -------
    >>> pit_mse = PitWrapper(nn.MSELoss(reduction="none"))
    >>> targets = torch.rand((2, 32, 4))
    >>> p = (3, 0, 2, 1)
    >>> predictions = targets[..., p]
    >>> loss, opt_p = pit_mse(predictions, targets)
    >>> loss
    tensor([0., 0.])
    """

    def __init__(self, base_loss):
        super(PitWrapper, self).__init__()
        self.base_loss = base_loss

    def _fast_pit(self, loss_mat):
        """
        Arguments
        ----------
        loss_mat : torch.Tensor
            Tensor of shape [sources, source] containing loss values for each
            possible permutation of predictions.

        Returns
        -------
        loss : torch.Tensor
            Permutation invariant loss for the current batch, tensor of shape [1]

        assigned_perm : tuple
            Indexes for optimal permutation of the input over sources which
            minimizes the loss.
        """

        loss = None
        assigned_perm = None
        for p in permutations(range(loss_mat.shape[0])):
            c_loss = loss_mat[range(loss_mat.shape[0]), p].mean()
            if loss is None or loss > c_loss:
                loss = c_loss
                assigned_perm = p
        return loss, assigned_perm

    def _opt_perm_loss(self, pred, target):
        """
        Arguments
        ---------
        pred : torch.Tensor
            Network prediction for the current example, tensor of
            shape [..., sources].
        target : torch.Tensor
            Target for the current example, tensor of shape [..., sources].

        Returns
        -------
        loss : torch.Tensor
            Permutation invariant loss for the current example, tensor of shape [1]

        assigned_perm : tuple
            Indexes for optimal permutation of the input over sources which
            minimizes the loss.

        """

        n_sources = pred.size(-1)

        pred = pred.unsqueeze(-2).repeat(
            *[1 for x in range(len(pred.shape) - 1)], n_sources, 1
        )
        target = target.unsqueeze(-1).repeat(
            1, *[1 for x in range(len(target.shape) - 1)], n_sources
        )

        loss_mat = self.base_loss(pred, target)
        assert (
            len(loss_mat.shape) >= 2
        ), "Base loss should not perform any reduction operation"
        mean_over = [x for x in range(len(loss_mat.shape))]
        loss_mat = loss_mat.mean(dim=mean_over[:-2])

        return self._fast_pit(loss_mat)

    def reorder_tensor(self, tensor, p):
        """
        Arguments
        ---------
        tensor : torch.Tensor
            Tensor to reorder given the optimal permutation, of shape
            [batch, ..., sources].
        p : list of tuples
            List of optimal permutations, e.g. for batch=2 and n_sources=3
            [(0, 1, 2), (0, 2, 1].

        Returns
        -------
        reordered : torch.Tensor
            Reordered tensor given permutation p.
        """

        reordered = torch.zeros_like(tensor, device=tensor.device)
        for b in range(tensor.shape[0]):
            reordered[b] = tensor[b][..., p[b]].clone()
        return reordered

    def forward(self, preds, targets):
        """
            Arguments
            ---------
            preds : torch.Tensor
                Network predictions tensor, of shape
                [batch, channels, ..., sources].
            targets : torch.Tensor
                Target tensor, of shape [batch, channels, ..., sources].

            Returns
            -------
            loss : torch.Tensor
                Permutation invariant loss for current examples, tensor of
                shape [batch]

            perms : list
                List of indexes for optimal permutation of the inputs over
                sources.
                e.g., [(0, 1, 2), (2, 1, 0)] for three sources and 2 examples
                per batch.
        """
        losses = []
        perms = []
        for pred, label in zip(preds, targets):
            loss, p = self._opt_perm_loss(pred, label)
            perms.append(p)
            losses.append(loss)
        loss = torch.stack(losses)
        return loss, perms





def compute_masked_loss(
    loss_fn,
    predictions,
    targets,
    length=None,
    label_smoothing=0.0,
    reduction="mean",
):
    """Compute the true average loss of a set of waveforms of unequal length.

    Arguments
    ---------
    loss_fn : function
        A function for computing the loss taking just predictions and targets.
        Should return all the losses, not a reduction (e.g. reduction="none").
    predictions : torch.Tensor
        First argument to loss function.
    targets : torch.Tensor
        Second argument to loss function.
    length : torch.Tensor
        Length of each utterance to compute mask. If None, global average is
        computed and returned.
    label_smoothing: float
        The proportion of label smoothing. Should only be used for NLL loss.
        Ref: Regularizing Neural Networks by Penalizing Confident Output
        Distributions. https://arxiv.org/abs/1701.06548
    reduction : str
        One of 'mean', 'batch', 'batchmean', 'none' where 'mean' returns a
        single value and 'batch' returns one per item in the batch and
        'batchmean' is sum / batch_size and 'none' returns all.
    """
    mask = torch.ones_like(targets)
    if length is not None:
        length_mask = length_to_mask(
            length * targets.shape[1], max_len=targets.shape[1],
        )

        # Handle any dimensionality of input
        while len(length_mask.shape) < len(mask.shape):
            length_mask = length_mask.unsqueeze(-1)
        length_mask = length_mask.type(mask.dtype)
        mask *= length_mask

    # Compute, then reduce loss
    loss = loss_fn(predictions, targets) * mask
    N = loss.size(0)
    if reduction == "mean":
        loss = loss.sum() / torch.sum(mask)
    elif reduction == "batchmean":
        loss = loss.sum() / N
    elif reduction == "batch":
        loss = loss.reshape(N, -1).sum(1) / mask.reshape(N, -1).sum(1)

    if label_smoothing == 0:
        return loss
    else:
        loss_reg = torch.mean(predictions, dim=1) * mask
        if reduction == "mean":
            loss_reg = torch.sum(loss_reg) / torch.sum(mask)
        elif reduction == "batchmean":
            loss_reg = torch.sum(loss_reg) / targets.shape[0]
        elif reduction == "batch":
            loss_reg = loss_reg.sum(1) / mask.sum(1)

        return -label_smoothing * loss_reg + (1 - label_smoothing) * loss


def get_si_snr_with_pitwrapper(source, estimate_source):
    """This function wraps si_snr calculation with the speechbrain pit-wrapper.

    Arguments:
    ---------
    source: [B, T, C],
        Where B is the batch size, T is the length of the sources, C is
        the number of sources the ordering is made so that this loss is
        compatible with the class PitWrapper.

    estimate_source: [B, T, C]
        The estimated source.

    Example:
    ---------
    >>> x = torch.arange(600).reshape(3, 100, 2)
    >>> xhat = x[:, :, (1, 0)]
    >>> si_snr = -get_si_snr_with_pitwrapper(x, xhat)
    >>> print(si_snr)
    tensor([135.2284, 135.2284, 135.2284])
    """

    pit_si_snr = PitWrapper(cal_si_snr)
    loss, perms = pit_si_snr(source, estimate_source)

    return loss


def cal_si_snr(source, estimate_source):
    """Calculate SI-SNR.

    Arguments:
    ---------
    source: [T, B, C],
        Where B is batch size, T is the length of the sources, C is the number of sources
        the ordering is made so that this loss is compatible with the class PitWrapper.

    estimate_source: [T, B, C]
        The estimated source.

    Example:
    ---------
    >>> import numpy as np
    >>> x = torch.Tensor([[1, 0], [123, 45], [34, 5], [2312, 421]])
    >>> xhat = x[:, (1, 0)]
    >>> x = x.unsqueeze(-1).repeat(1, 1, 2)
    >>> xhat = xhat.unsqueeze(1).repeat(1, 2, 1)
    >>> si_snr = -cal_si_snr(x, xhat)
    >>> print(si_snr)
    tensor([[[ 25.2142, 144.1789],
             [130.9283,  25.2142]]])
    """
    EPS = 1e-8
    assert source.size() == estimate_source.size()
    device = estimate_source.device.type

    source_lengths = torch.tensor(
        [estimate_source.shape[0]] * estimate_source.shape[1], device=device
    )
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    num_samples = (
        source_lengths.contiguous().reshape(1, -1, 1).float()
    )  # [1, B, 1]
    mean_target = torch.sum(source, dim=0, keepdim=True) / num_samples
    mean_estimate = (
        torch.sum(estimate_source, dim=0, keepdim=True) / num_samples
    )
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = zero_mean_target  # [T, B, C]
    s_estimate = zero_mean_estimate  # [T, B, C]
    # s_target = <s', s>s / ||s||^2
    dot = torch.sum(s_estimate * s_target, dim=0, keepdim=True)  # [1, B, C]
    s_target_energy = (
        torch.sum(s_target ** 2, dim=0, keepdim=True) + EPS
    )  # [1, B, C]
    proj = dot * s_target / s_target_energy  # [T, B, C]
    # e_noise = s' - s_target
    e_noise = s_estimate - proj  # [T, B, C]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    si_snr_beforelog = torch.sum(proj ** 2, dim=0) / (
        torch.sum(e_noise ** 2, dim=0) + EPS
    )
    si_snr = 10 * torch.log10(si_snr_beforelog + EPS)  # [B, C]

    return -si_snr.unsqueeze(0)


def get_mask(source, source_lengths):
    """
    Arguments
    ---------
    source : [T, B, C]
    source_lengths : [B]

    Returns
    -------
    mask : [T, B, 1]

    Example:
    ---------
    >>> source = torch.randn(4, 3, 2)
    >>> source_lengths = torch.Tensor([2, 1, 4]).int()
    >>> mask = get_mask(source, source_lengths)
    >>> print(mask)
    tensor([[[1.],
             [1.],
             [1.]],
    <BLANKLINE>
            [[1.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]]])
    """
    T, B, _ = source.size()
    mask = source.new_ones((T, B, 1))
    for i in range(B):
        mask[source_lengths[i] :, i, :] = 0
    return mask






if __name__ == "__main__":
    s1 = librosa.load('s1.wav', sr=8000)[0]
    s2 = librosa.load('s2.wav', sr=8000)[0]
    source = np.stack((s1, s2), axis=1)
    estimate_source = np.stack((s2 * 1, s1 * 1), axis=1)
    source = torch.from_numpy(np.expand_dims(source, axis=0))
    estimate_source = torch.from_numpy(np.expand_dims(estimate_source, axis=0))

    source_lengths = torch.FloatTensor([40000, 40000]).type(torch.int)
    print('source', source.shape)
    print('estimate_source', estimate_source.shape)
    print('source_lengths', source_lengths)

    loss = -get_si_snr_with_pitwrapper(source, estimate_source)
    print('loss', loss)
    # print('max_snr', max_snr)
    # print('reorder_estimate_source', reorder_estimate_source.shape)

