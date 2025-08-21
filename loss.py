import mlx.core as mx


class HardNegativeInfoNCELoss:
    def __init__(self, temperature: float = 0.1, beta: float = 1.0):
        self.temperature = temperature
        self.beta = beta

    def __call__(
        self,
        query_embeddings: mx.array,
        doc_embeddings: mx.array,
        weights: mx.array | None = None,
        unnormalized_loss: bool = False,
    ) -> mx.array:
        eps = 1e-8
        logits = mx.matmul(query_embeddings, doc_embeddings.T) / self.temperature
        n = logits.shape[0]
        idx = mx.arange(n, dtype=mx.int32)
        pos_scores = logits[idx, idx]

        # Masks
        pos_mask = mx.eye(n)
        neg_mask = 1 - pos_mask

        # Numerical stability via row-wise max
        row_max = mx.max(logits, axis=1, keepdims=True)
        pos_exp = mx.exp(pos_scores - mx.squeeze(row_max, axis=1))
        neg_exp = mx.exp(logits - row_max) * neg_mask

        if self.beta > 0:
            # Importance weights on negatives; normalize to mean 1 per-row
            imp = mx.power(neg_exp + eps, self.beta)
            mean_imp = mx.sum(imp, axis=1, keepdims=True) / (
                mx.sum(neg_mask, axis=1, keepdims=True) + eps
            )
            imp_norm = imp / (mean_imp + eps)
            neg_sum = mx.sum(neg_exp * imp_norm, axis=1)
        else:
            neg_sum = mx.sum(neg_exp, axis=1)

        loss = -mx.log(pos_exp / (pos_exp + neg_sum + eps))
        if unnormalized_loss:
            return loss

        if weights is not None:
            w = weights.astype(loss.dtype)
            return mx.sum(loss * w) / (mx.sum(w) + eps)
        else:
            return mx.mean(loss)


class InfoNCELoss:
    def __init__(self, temperature: float = 0.05):
        self.temperature = temperature

    def __call__(
        self,
        query_embeddings: mx.array,
        doc_embeddings: mx.array,
        weights: mx.array | None = None,
        unnormalized_loss: bool = False,
    ) -> mx.array:
        logits = mx.matmul(query_embeddings, doc_embeddings.T) / self.temperature

        max_row = mx.max(logits, axis=1, keepdims=True)
        lse_row = mx.squeeze(max_row, axis=1) + mx.log(
            mx.sum(mx.exp(logits - max_row), axis=1)
        )

        max_col = mx.max(logits, axis=0, keepdims=True)
        lse_col = mx.squeeze(max_col, axis=0) + mx.log(
            mx.sum(mx.exp(logits - max_col), axis=0)
        )

        n = logits.shape[0]
        idx = mx.arange(n, dtype=mx.int32)
        pos = logits[idx, idx]

        loss_i2t = -pos + lse_row
        loss_t2i = -pos + lse_col
        per_example = 0.5 * (loss_i2t + loss_t2i)

        if unnormalized_loss:
            return per_example

        if weights is not None:
            w = weights.astype(per_example.dtype)
            return mx.sum(per_example * w) / (mx.sum(w) + 1e-8)
        else:
            return mx.mean(per_example)


class NTXentLossWithAdvancedMining:
    """
    Advanced NT-Xent loss with positive-aware hard negative mining.
    Based on research from NV-Retriever and other SOTA embedding models.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        mining_method: str = "perc_pos",
        mining_margin: float = 0.95,
        num_hard_negatives: int = 4,
    ):
        self.temperature = temperature
        self.mining_method = mining_method
        self.mining_margin = mining_margin
        self.num_hard_negatives = num_hard_negatives

    def __call__(
        self,
        query_embeddings: mx.array,
        doc_embeddings: mx.array,
        weights: mx.array | None = None,
        unnormalized_loss: bool = False,
    ) -> mx.array:
        eps = 1e-8

        logits = mx.matmul(query_embeddings, doc_embeddings.T) / self.temperature
        n = logits.shape[0]

        idx = mx.arange(n, dtype=mx.int32)
        pos_scores = logits[idx, idx]

        pos_mask = mx.eye(n)
        neg_mask = 1 - pos_mask

        if self.mining_method == "perc_pos":
            # Keep negatives that are close to positives (hard): score >= margin * pos
            neg_threshold = pos_scores.reshape(-1, 1) * self.mining_margin
            valid_neg_mask = (logits >= neg_threshold) * neg_mask
        elif self.mining_method == "margin_pos":
            # Keep negatives within a margin of the positive score: score >= pos - margin
            neg_threshold = pos_scores.reshape(-1, 1) - self.mining_margin
            valid_neg_mask = (logits >= neg_threshold) * neg_mask
        else:
            valid_neg_mask = neg_mask

        # Optional: per-row top-k hard negatives selection
        if self.num_hard_negatives is not None and self.num_hard_negatives > 0:
            k = int(self.num_hard_negatives)
            # Initialize selection
            selected_mask = mx.zeros_like(valid_neg_mask)
            available_mask = valid_neg_mask
            very_small = mx.full_like(logits, -1e9)
            for _ in range(k):
                # Set unavailable entries to very small so they won't be selected
                effective_scores = logits * available_mask + very_small * (
                    1 - available_mask
                )
                row_top = mx.max(effective_scores, axis=1, keepdims=True)
                step_select = (effective_scores == row_top) * available_mask
                selected_mask = mx.maximum(selected_mask, step_select)
                # Remove selected for next iteration
                available_mask = available_mask * (1 - step_select)
            valid_neg_mask = selected_mask

        # Numerical stability via row-wise max
        row_max = mx.max(logits, axis=1, keepdims=True)
        pos_exp = mx.exp(pos_scores - mx.squeeze(row_max, axis=1))
        neg_exp = mx.exp(logits - row_max) * valid_neg_mask

        neg_sum = mx.sum(neg_exp, axis=1)

        loss = -mx.log(pos_exp / (pos_exp + neg_sum + eps))

        if unnormalized_loss:
            return loss

        if weights is not None:
            w = weights.astype(loss.dtype)
            return mx.sum(loss * w) / (mx.sum(w) + eps)
        else:
            return mx.mean(loss)
