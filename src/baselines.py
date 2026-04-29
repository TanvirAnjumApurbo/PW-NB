"""Baseline NB classifiers and the PW-NB registry."""

from __future__ import annotations

import logging
from typing import Callable

from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler, StandardScaler

from src.pw_nb import AdaptivePWNB, GaussianPWNB

logger = logging.getLogger("pwnb.baselines")


def get_baselines(
    k_values: list[int] | None = None,
    random_state: int = 42,
) -> dict[str, Callable[[], ClassifierMixin]]:
    """Return factory functions for all classifiers.

    Parameters
    ----------
    k_values : list of int, default=[5, 15, 30, 45]
        k values for PW-NB variants.
    random_state : int
        Random state for PW-NB.

    Returns
    -------
    dict mapping classifier name to a factory function.
    """
    if k_values is None:
        k_values = [5, 15, 30, 45]

    registry: dict[str, Callable[[], ClassifierMixin]] = {
        "GaussianNB": lambda: GaussianNB(),
        "MultinomialNB": lambda: Pipeline(
            [
                ("scale", MinMaxScaler()),
                ("clf", MultinomialNB()),
            ]
        ),
        "BernoulliNB": lambda: Pipeline(
            [
                ("scale", StandardScaler()),
                ("bin", Binarizer(threshold=0.0)),
                ("clf", BernoulliNB()),
            ]
        ),
        "ComplementNB": lambda: Pipeline(
            [
                ("scale", MinMaxScaler()),
                ("clf", ComplementNB()),
            ]
        ),
    }

    for k in k_values:
        kk = k  # capture
        registry[f"PW-NB(k={kk})"] = lambda kk=kk: GaussianPWNB(
            k=kk, random_state=random_state
        )

    registry["PW-NB(auto)"] = lambda: AdaptivePWNB(
        k_candidates=tuple(k_values),
        inner_folds=3,
        random_state=random_state,
    )

    return registry


def adapt_k_for_dataset(k: int, y: list | object, dataset_name: str) -> int:
    """Adapt k to min(k, n_min_class - 1) for small datasets."""
    import numpy as np

    _, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    if k >= min_count:
        new_k = max(1, min_count - 1)
        logger.info(
            "%s: adapted k=%d to k=%d (min class size=%d)",
            dataset_name,
            k,
            new_k,
            min_count,
        )
        return new_k
    return k
