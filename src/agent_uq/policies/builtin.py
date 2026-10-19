from __future__ import annotations

import random
from typing import Sequence

from agent_uq.policies.base import SelectableCandidate, register_policy


@register_policy("random")
class RandomPolicy:


    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)


        self._rng = random.Random(self.seed)

    def choose(self, selectable: Sequence[SelectableCandidate]) -> int:
        if not selectable:
            raise ValueError("RandomPolicy.choose: no parse-valid candidates")


        rng = random.Random(self.seed + len(selectable) * 7919 + sum(s.raw_index for s in selectable))

        chosen = rng.choice(list(selectable))
        return chosen.raw_index
