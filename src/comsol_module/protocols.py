from pathlib import Path
from typing import Protocol

import numpy as np

from .comsol_meta import ComsolMetaData


class ModuleClass(Protocol):
    metadata: ComsolMetaData
    path: Path | None
    name: str

    @property
    def time_vales(self) -> np.ndarray: ...

    @property
    def exported_field(self) -> list[str]: ...
