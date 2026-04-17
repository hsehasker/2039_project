"""Deterministic dataset sharding for a SISA-style ensemble.

SISA (Bourtoule et al., *"Machine Unlearning"*, IEEE S&P 2021) partitions the
training set into ``S`` disjoint shards. Each shard trains an independent
"slice" model; predictions are aggregated (e.g. soft voting).

When a user requests deletion of record ``r``:

  * only the shard that contained ``r`` needs to be re-trained;
  * the other ``S - 1`` shards remain untouched;
  * expected re-training cost is ``1/S`` of a full retrain.

This module implements the *sharding bookkeeping* only: the split itself
(which IDs go to which shard) is persisted so that after unlearning we can
recompute the same partition minus the deleted rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def assign_shards(
    user_ids: Sequence[str] | np.ndarray,
    n_shards: int,
    seed: int = 0,
) -> np.ndarray:
    """Hash user IDs deterministically into ``n_shards`` buckets.

    Using a stable hash keyed by ``seed`` means that a user always lands in
    the same shard regardless of row ordering — this is critical for making
    delete-by-ID requests O(1) to route.
    """
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    user_ids = np.asarray(user_ids)
    shard = np.empty(len(user_ids), dtype=np.int64)
    for i, uid in enumerate(user_ids):
        # Deterministic FNV-1a 64-bit hash of "seed::uid".
        # numpy's internal hashing is not stable across versions; Python's
        # ``hash()`` is process-salted. FNV-1a gives us cross-run, cross-python
        # stability which is essential for consistent shard routing.
        key = f"{seed}::{uid}".encode("utf-8")
        h = 0xcbf29ce484222325
        for b in key:
            h ^= b
            h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        shard[i] = h % n_shards
    return shard


@dataclass
class ShardPlan:
    """Per-shard row indices into the training dataframe."""
    n_shards: int
    shard_of_row: np.ndarray  # shape (n_rows,), values in [0, n_shards)
    indices: list[np.ndarray]  # indices[i] -> array of row ids in shard i

    @classmethod
    def build(cls, user_ids: Sequence[str], n_shards: int, seed: int = 0) -> "ShardPlan":
        shard_of_row = assign_shards(user_ids, n_shards, seed=seed)
        indices = [np.where(shard_of_row == s)[0] for s in range(n_shards)]
        return cls(n_shards=n_shards, shard_of_row=shard_of_row, indices=indices)

    def shard_for(self, user_id: str, seed: int = 0) -> int:
        return int(assign_shards([user_id], self.n_shards, seed=seed)[0])

    def sizes(self) -> list[int]:
        return [int(len(idx)) for idx in self.indices]

    def rebuild_after_deletion(
        self,
        deleted_rows: Iterable[int],
    ) -> tuple["ShardPlan", set[int]]:
        """Return a new plan with ``deleted_rows`` removed + list of affected shards."""
        deleted = set(int(x) for x in deleted_rows)
        affected = {int(self.shard_of_row[r]) for r in deleted if 0 <= r < len(self.shard_of_row)}
        new_indices = [
            np.array([r for r in shard_rows if r not in deleted], dtype=np.int64)
            for shard_rows in self.indices
        ]
        keep_mask = np.array([r not in deleted for r in range(len(self.shard_of_row))])
        new_shard_of_row = self.shard_of_row[keep_mask]
        # Row indices shift after deletion; remap to new positions
        remap = np.cumsum(keep_mask.astype(int)) - 1
        new_indices = [remap[idx] for idx in new_indices]
        return ShardPlan(self.n_shards, new_shard_of_row, new_indices), affected
