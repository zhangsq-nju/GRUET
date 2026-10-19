import numpy as np

from abc import ABC, abstractmethod
from typing import List, Dict
from lm_polygraph.utils.common import polygraph_module_init


class Estimator(ABC):


    @polygraph_module_init
    def __init__(self, stats_dependencies: List[str], level: str):

        assert level in ["sequence", "claim", "token"]
        self.level = level
        self.stats_dependencies = stats_dependencies

    @abstractmethod
    def __str__(self):

        raise Exception("Not implemented")

    @abstractmethod
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:

        raise Exception("Not implemented")
