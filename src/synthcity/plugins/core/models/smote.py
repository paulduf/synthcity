# stdlib
from typing import Any, Optional, Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments


class SMOTE:
    """
    SMOTE model
    """
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        X: pd.DataFrame,
        n_neighbours = 5
    ):
        # super(SMOTE, self).__init__(  # TODO: inheritance needed ?
        pass

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.DataFrame) -> "SMOTE":
        pass

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: int,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        pass


