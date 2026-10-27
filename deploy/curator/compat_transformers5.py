# Copyright 2022 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Backblaze, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Embed1 tensor helpers adapted from Transformers 4.57.6 for version 5.

Only model math is retained; no legacy model-loading or serialization code is used.
"""

import torch


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        index = head - sum(pruned < head for pruned in already_pruned_heads)
        mask[index] = 0
    mask = mask.view(-1).contiguous().eq(1)
    return heads, torch.arange(len(mask))[mask].long()


class BertMaskMixin:
    """Preserve the custom Q-Former's attention-mask semantics."""

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        if head_mask.dim() == 1:
            head_mask = head_mask[None, None, :, None, None].expand(
                num_hidden_layers, -1, -1, -1, -1
            )
        elif head_mask.dim() == 2:
            head_mask = head_mask[:, None, :, None, None]
        if head_mask.dim() != 5:
            raise ValueError("head_mask must expand to five dimensions")
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask.unsqueeze(-1) if is_attention_chunked else head_mask

    def invert_attention_mask(self, encoder_attention_mask):
        if encoder_attention_mask.dim() == 3:
            mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError("encoder_attention_mask must have two or three dimensions")
        return (1.0 - mask.to(dtype=self.dtype)) * torch.finfo(self.dtype).min


@torch.no_grad()
def restore_video_normalization(model):
    """Restore nonpersistent constants after Transformers materializes meta buffers."""
    for name, values in (
        ("normalization_mean", (0.485, 0.456, 0.406)),
        ("normalization_std", (0.229, 0.224, 0.225)),
    ):
        buffer = getattr(model, name)
        buffer.copy_(buffer.new_tensor(values).view_as(buffer))
    return model
