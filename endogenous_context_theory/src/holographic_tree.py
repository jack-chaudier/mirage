"""Holographic Context Tree (binary carry forest) over tropical elements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .tropical_semiring import Event, NEG_INF, TropicalContext, compose_tropical


@dataclass
class TreeNode:
    context: TropicalContext
    left: Optional["TreeNode"]
    right: Optional["TreeNode"]
    start_eid: int
    end_eid: int
    height: int

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class HolographicContextTree:
    """Fenwick/binomial-style forest preserving temporal composition order."""

    def __init__(self, k: int):
        self.k = k
        self.forest: List[Optional[TreeNode]] = []
        self.count = 0

    def _ensure_height(self, h: int) -> None:
        while len(self.forest) <= h:
            self.forest.append(None)

    def append(self, event: Event) -> None:
        node = TreeNode(
            context=TropicalContext.from_event(event, self.k),
            left=None,
            right=None,
            start_eid=event.eid,
            end_eid=event.eid,
            height=0,
        )
        h = 0
        while True:
            self._ensure_height(h)
            if self.forest[h] is None:
                self.forest[h] = node
                break

            existing = self.forest[h]
            self.forest[h] = None
            assert existing is not None

            # existing is earlier, node is later.
            merged = compose_tropical(existing.context, node.context)
            node = TreeNode(
                context=merged,
                left=existing,
                right=node,
                start_eid=existing.start_eid,
                end_eid=node.end_eid,
                height=h + 1,
            )
            h += 1

        self.count += 1

    def _trees_temporal_order(self) -> List[TreeNode]:
        # Higher indices contain earlier events due carry propagation.
        trees: List[TreeNode] = []
        for h in range(len(self.forest) - 1, -1, -1):
            node = self.forest[h]
            if node is not None:
                trees.append(node)
        return trees

    def get_root_summary(self) -> TropicalContext:
        acc = TropicalContext.empty(self.k)
        for tree in self._trees_temporal_order():
            acc = compose_tropical(acc, tree.context)
        return acc

    def forest_size(self) -> int:
        return sum(1 for n in self.forest if n is not None)

    def depth(self) -> int:
        if not self.forest:
            return 0
        for h in range(len(self.forest) - 1, -1, -1):
            if self.forest[h] is not None:
                return h + 1
        return 0

    def _best_feasible_shifted(self, ctx: TropicalContext, prefix_d: int) -> float:
        if prefix_d >= self.k:
            return float(np.max(ctx.W))
        threshold = self.k - prefix_d
        if threshold < 0:
            threshold = 0
        return float(np.max(ctx.W[threshold:]))

    @staticmethod
    def _eq(a: float, b: float, tol: float = 1e-12) -> bool:
        if np.isneginf(a) and np.isneginf(b):
            return True
        return abs(a - b) <= tol

    def _descend_for_target(
        self,
        node: TreeNode,
        prefix_d: int,
        target_weight: float,
    ) -> Optional[TreeNode]:
        if node.is_leaf:
            return node if self._eq(self._best_feasible_shifted(node.context, prefix_d), target_weight) else None

        assert node.left is not None
        assert node.right is not None

        left = node.left
        right = node.right
        left_best = self._best_feasible_shifted(left.context, prefix_d)
        right_best = self._best_feasible_shifted(right.context, prefix_d + left.context.d_total)

        if self._eq(left_best, target_weight) and left_best >= right_best - 1e-12:
            return self._descend_for_target(left, prefix_d, target_weight)
        if self._eq(right_best, target_weight):
            return self._descend_for_target(right, prefix_d + left.context.d_total, target_weight)
        if self._eq(left_best, target_weight):
            return self._descend_for_target(left, prefix_d, target_weight)
        return None

    def find_pivot_block(self) -> Optional[TreeNode]:
        root = self.get_root_summary()
        target = root.W[self.k]
        if np.isneginf(target):
            return None

        prefix = 0
        for tree in self._trees_temporal_order():
            best = self._best_feasible_shifted(tree.context, prefix)
            if self._eq(best, target):
                found = self._descend_for_target(tree, prefix, target)
                if found is not None:
                    return found
            prefix += tree.context.d_total
        return None
