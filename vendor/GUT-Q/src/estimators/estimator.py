from abc import ABC, abstractmethod
from typing import List, Dict
from src.utils.process_prob_info import standard_probinfo_weight

class Estimator(ABC):


    @classmethod
    def get_subclass_dict(cls):

        return {subclass.__name__: subclass for subclass in cls.__subclasses__()}


    def __init__(self, **kwargs):

        self.merge_way = kwargs.get("merge_way", "aov")
        if self.merge_way != "aov":
            raise ValueError("Only merge_way='aov' is supported")
        self.sim_way = kwargs.get("sim_way", "2entail")

        self.weight = kwargs.get("weight", "")
        if self.weight not in ("", "prob"):
            raise ValueError("weight must be empty (unweighted) or 'prob'")
        self.token_level = kwargs.get("token_level", "-")
        self.step_aggregation = kwargs.get("step_aggregation", "-")
        self.beta = kwargs.get("beta", 1)

        if self.weight == 'prob':
            if self.token_level == '-':
                raise ValueError("token_level must be specified when weight is 'prob'")
            if self.step_aggregation == '-':
                raise ValueError("step_aggregation must be specified when weight is 'prob'")

        self.window_size = kwargs.get("window_size", 6)

    @abstractmethod
    def __str__(self):

        raise Exception("Not implemented")

    @abstractmethod
    def __call__(self, g_data) -> List[float]:

        raise Exception("Not implemented")

    def _get_weight(self, node):

        if self.weight == 'prob':
            v = sum(
                standard_probinfo_weight(prob_info_dict, self.token_level, way=self.step_aggregation, window_size=self.window_size)
                for prob_info_dict in node['prob_info_all']
            )/len(node['prob_info_all']) if 'prob_info_all' in node else 0
            return v + self.beta
        else:
            return 1
