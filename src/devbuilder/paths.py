from __future__ import annotations

import os
from pathlib import Path


class PathManager:
    def __init__(self, base_dir: Path | str | None = None):
        root = base_dir if base_dir is not None else os.getenv("DEVBUILDER_ROOT", ".")
        self.base_dir = Path(root).resolve()

    @property
    def configs(self) -> Path:
        return self.base_dir / "configs"

    @property
    def data(self) -> Path:
        return self.base_dir / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def runs(self) -> Path:
        return self.base_dir / "runs"

    @property
    def artifacts(self) -> Path:
        return self.base_dir / "artifacts"

    @property
    def models(self) -> Path:
        return self.artifacts / "models"

    def ensure_dirs(self) -> None:
        """
        Create all directories if they do not exist.
        """
        for prop in [
            self.configs,
            self.raw,
            self.processed,
            self.models,
            self.runs,
        ]:
            prop.mkdir(parents=True, exist_ok=True)


# Default instance for convenience
paths = PathManager()
