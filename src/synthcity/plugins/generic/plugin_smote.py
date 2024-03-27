# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import Distribution, IntegerDistribution
from synthcity.plugins.core.models.smote import SMOTE
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class SMOTESamplerPlugin(Plugin):
    """
    TODO:
        - implement core
        - implement cond argument
        - implement hyperparameter optimization

    Called SMOTE Sampler to differentiate with genuine SMOTE used for imbalanced learning scenario.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(sampling_strategy="smote", **kwargs)

    @staticmethod
    def name() -> str:
        return "smote_sampler"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        raise NotImplementedError
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "SMOTESamplerPlugin":

        self.model = SMOTE()
        self.model.fit(X.dataframe())

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.generate, count, syn_schema)


plugin = SMOTESamplerPlugin
