"""Central filesystem configuration. No scientific settings or automatic writes."""
from dataclasses import dataclass
import os
from pathlib import Path
import tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ENV = "WL_PATHS_CONFIG"


@dataclass(frozen=True)
class ProjectPaths:
    data_root: Path
    work_root: Path
    glimpse_root: Path
    mcalens_root: Path
    repository_root: Path = REPOSITORY_ROOT

    @property
    def code_root(self):
        return self.repository_root / "code" / "pipeline"

    @property
    def prior_file(self):
        return self.repository_root / "config" / "priors" / "fair_validation_pk_mean.npy"

    @property
    def glimpse_executable(self):
        return self.glimpse_root / "build" / "glimpse"

    def as_dict(self):
        return {name: str(getattr(self, name)) for name in (
            "data_root", "work_root", "glimpse_root", "mcalens_root",
            "repository_root", "code_root", "prior_file", "glimpse_executable")}


def load_paths(config_file=None):
    config_file = Path(config_file or os.environ.get(CONFIG_ENV) or
                       REPOSITORY_ROOT / "config_paths.toml").expanduser().resolve()
    with config_file.open("rb") as stream:
        settings = tomllib.load(stream)
    names = {"data_root", "work_root", "glimpse_root", "mcalens_root"}
    if set(settings) != names:
        raise ValueError(f"{config_file}: expected exactly {sorted(names)}; got {sorted(settings)}")
    resolved = {}
    for name, value in settings.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{config_file}: {name} must be a non-empty path string")
        path = Path(value).expanduser()
        resolved[name] = (path if path.is_absolute() else config_file.parent / path).resolve()
    # Existing runners create outputs at import time. Reject work roots that
    # would mix those writes with the archival inputs or source tree.
    work = resolved["work_root"]
    protected = [resolved["data_root"], REPOSITORY_ROOT / "results", REPOSITORY_ROOT / "config",
                 REPOSITORY_ROOT / "provenance", REPOSITORY_ROOT / "code"]
    if work == REPOSITORY_ROOT or any(work.is_relative_to(p) or p.is_relative_to(work) for p in protected):
        raise ValueError("work_root must be separate from raw data and this repository's archived inputs/source")
    return ProjectPaths(**resolved)


PATHS = load_paths()
