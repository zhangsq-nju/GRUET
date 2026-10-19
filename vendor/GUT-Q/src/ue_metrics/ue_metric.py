import numpy as np

from typing import List
from abc import ABC, abstractmethod


def normalize(target: List[float]):
    min_t, max_t = np.min(target), np.max(target)
    if np.isclose(min_t, max_t):
        min_t -= 1
        max_t += 1
    target = (np.array(target) - min_t) / (max_t - min_t)
    return target


def skip_target_nans(target, estimator):
    newt, newe = [], []
    for t, e in zip(target, estimator):
        if np.isnan(t):
            continue
        newt.append(t)
        newe.append(e)
    return newt, newe


class UEMetric(ABC):


    @abstractmethod
    def __str__(self):

        raise Exception("Not implemented")

    @abstractmethod
    def __call__(self, estimator: List[float], target: List[float]) -> float:

        raise Exception("Not implemented")


def get_random_scores(function, metrics, num_iter=1000, seed=42):
    np.random.seed(seed)
    rand_scores = np.arange(len(metrics))

    value = []
    for i in range(num_iter):
        np.random.shuffle(rand_scores)
        rand_val = function(rand_scores, metrics)
        value.append(rand_val)
    return np.mean(value)


def normalize_metric(target_score, oracle_score, random_score):
    if not (oracle_score == random_score):
        target_score = (target_score - random_score) / (oracle_score - random_score)
    return target_score
