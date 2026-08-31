from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from .comsol_meta import ComsolMetaData
from .protocols import ModuleClass


@dataclass
class ComsolCsv(ModuleClass):
    path: Path
    data: pd.DataFrame
    metadata: ComsolMetaData = field(default_factory=ComsolMetaData)
    name: str = ""

    @classmethod
    def from_csv(cls, path: Path | str) -> Self:
        data = pd.read_csv(path, skiprows=4, index_col=0)

        return cls(
            data=data,
            path=Path(path),
            metadata=ComsolMetaData(
                exported_fields=list(data.columns),
            ),
        )

    @property
    def time_values(self) -> np.ndarray:
        return np.asarray(self.data.index)

    @property
    def exported_fields(self) -> list[str]:
        return list(self.data.columns)
