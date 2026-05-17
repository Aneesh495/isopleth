"""Multi-horizon training curriculum and scheduled sampling strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

@dataclass
class CurriculumStage:
    """A stage in multi-horizon curriculum training."""
    max_epoch: int
    rollout_horizon: int
    teacher_forcing_ratio: float

class CurriculumScheduler:
    """Manages horizon length expansion and scheduled sampling decay across training epochs."""

    def __init__(self, stages: Optional[Sequence[CurriculumStage]] = None) -> None:
        if stages is None:
            # Default progressive curriculum: 1 step -> 2 steps -> 4 steps -> 8 steps
            self.stages = [
                CurriculumStage(max_epoch=5, rollout_horizon=1, teacher_forcing_ratio=1.0),
                CurriculumStage(max_epoch=10, rollout_horizon=2, teacher_forcing_ratio=0.75),
                CurriculumStage(max_epoch=15, rollout_horizon=4, teacher_forcing_ratio=0.50),
                CurriculumStage(max_epoch=30, rollout_horizon=8, teacher_forcing_ratio=0.20),
            ]
        else:
            self.stages = list(stages)

    def get_stage(self, epoch: int) -> CurriculumStage:
        """Retrieve active curriculum stage for current epoch."""
        for stage in self.stages:
            if epoch <= stage.max_epoch:
                return stage
        return self.stages[-1]

    def get_rollout_horizon(self, epoch: int) -> int:
        return self.get_stage(epoch).rollout_horizon

    def get_teacher_forcing_ratio(self, epoch: int) -> float:
        return self.get_stage(epoch).teacher_forcing_ratio
