
from abc import ABCMeta, abstractmethod
from collections.abc import Callable

from ray_optim.target import Target


class OptimizerBackend(metaclass=ABCMeta):
    @abstractmethod
    def optimize(
        self,
        objective: Callable,
        iterations: int,
        target: Target,
        starting_point: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        pass