from pydantic import BaseModel
from typing import List
import numpy as np


class TransactionInput(BaseModel):
    features: List[float]  # expects a list like [0.1, 23.0, ...]

    def to_numpy(self) -> np.ndarray:
        # self.features: list of features -> np array
        return np.array(self.features, dtype=np.float32)
