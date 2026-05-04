"""Meta information about the application"""

import importlib.metadata

name = "comsol_module"
version = "?"
summary = "?"

try:
    metadata = importlib.metadata.metadata(name)
    version = metadata["Version"]
    summary = metadata["Summary"]
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    pass

print(name)
