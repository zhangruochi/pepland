"""Shared repository-relative paths and explicit node-index configuration."""
from pathlib import Path
from typing import Any, List, Optional, Union


def resolve_path(value: Union[str, Path], repo_root: Path, legacy_base: Optional[Path] = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [repo_root / path]
        if legacy_base is not None:
            candidates.append(legacy_base / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Configured path does not exist: {}".format(value))


def atom_index(value: Any) -> Optional[Union[int, List[int]]]:
    if value is None or value is False:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple)) or type(value).__name__ == 'ListConfig':
        values = list(value)
        if values and all(isinstance(i, int) and not isinstance(i, bool) for i in values):
            return values
    raise ValueError("atom_index must be false, null, an integer or a nonempty integer list")
