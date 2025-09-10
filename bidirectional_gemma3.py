#!/usr/bin/env python3
"""
Monkey patch for Gemma3 to support bidirectional attention.
"""

import mlx.core as mx
from mlx_lm.models.gemma3_text import Gemma3Model, create_attention_mask

# Global state to track patching
_original_call = None
_is_patched = False


def patch_gemma3_for_bidirectional():
    """Monkey patch the original Gemma3Model to support bidirectional attention."""
    global _original_call, _is_patched

    if _is_patched:
        return  # Already patched

    _original_call = Gemma3Model.__call__

    def patched_call(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        input_embeddings: mx.array = None,
        bidirectional: bool = False,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        h *= mx.array(self.args.hidden_size**0.5, mx.bfloat16).astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            j = self.args.sliding_window_pattern
            full_mask = create_attention_mask(h, cache[j - 1 : j])
            sliding_window_mask = create_attention_mask(h, cache)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_global = (
                i % self.args.sliding_window_pattern
                == self.args.sliding_window_pattern - 1
            )

            local_mask = mask
            if mask is None and is_global:
                local_mask = full_mask
            elif mask is None:
                local_mask = sliding_window_mask

            if bidirectional:
                h = layer(h, None, c)
            else:
                h = layer(h, local_mask, c)

        return self.norm(h)

    # Apply the patch
    Gemma3Model.__call__ = patched_call
    _is_patched = True


def unpatch_gemma3():
    """Restore the original Gemma3Model.__call__ method."""
    global _original_call, _is_patched

    if not _is_patched or _original_call is None:
        return  # Not patched or already restored

    Gemma3Model.__call__ = _original_call
    _is_patched = False
    _original_call = None
