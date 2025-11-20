# lab_uw/directory_manager.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

PathLike = Union[str, Path]

def _env_base_dir() -> Optional[Path]:
    v = os.getenv("LAB_UW_BASE")
    return Path(v).expanduser().resolve() if v else None

@dataclass(frozen=True)
class ProjectPaths:
    base_dir: Path

    # ---- roots ----
    def machine_root(self, machine: str) -> Path:
        return self.base_dir / f"experiments_{machine}"

    def experiment_root(self, machine: str, experiment: str) -> Path:
        return self.machine_root(machine) / experiment

    # ---- canonical subtrees ----
    def data_dir(self, machine: str, experiment: str, rel: PathLike) -> Path:
        return self.experiment_root(machine, experiment) / Path(rel)

    def analysis_root(self, machine: str, experiment: str) -> Path:
        return self.experiment_root(machine, experiment) / "data_analysis"

    def images_root(self, machine: str, experiment: str) -> Path:
        return self.experiment_root(machine, experiment) / "standard_images"

    # ---- convenient builders ----
    def ensure_dirs(self, *paths: PathLike) -> List[Path]:
        out: List[Path] = []
        for p in paths:
            pp = Path(p)
            pp.mkdir(parents=True, exist_ok=True)
            out.append(pp)
        return out

    def list_files(
        self,
        folder: PathLike,
        glob: str = "*",
        exts: Optional[Sequence[str]] = None,
    ) -> List[Path]:
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Directory not found: {folder}")
        items = sorted(folder.glob(glob))
        files = [p for p in items if p.is_file()]
        if exts:
            exts_lc = {e.lower() for e in exts}
            files = [p for p in files if p.suffix.lower() in exts_lc]
        return files


class DirectoryManager:
    """
    Backwards-compatible façade around ProjectPaths.

    Default precedence for base directory:
      1) Explicit 'base_dir' argument
      2) $LAB_UW_BASE environment variable
      3) parent of current working directory (Path.cwd().parent)
         -> matches your original behavior when running from lab_UW/
    """
    def __init__(self, base_dir: Optional[PathLike] = None):
        if base_dir is not None:
            base = Path(base_dir).expanduser().resolve()
        else:
            base = _env_base_dir() or Path.cwd().parent.resolve()
        self.paths = ProjectPaths(base)

    # --- backward-compat attribute (so code can use dir_manager.base_dir) ---
    @property
    def base_dir(self) -> Path:
        return self.paths.base_dir

    # --- helpers unchanged ---
    def make_infile_path_list(self, machine_name: str, experiment_name: str, data_type: str) -> List[Path]:
        indir_path = self.paths.data_dir(machine_name, experiment_name, data_type)
        return self.paths.list_files(indir_path)

    def make_data_analysis_folders(self, machine_name: str, experiment_name: str, data_types: List[str]) -> List[Path]:
        root = self.paths.analysis_root(machine_name, experiment_name)
        outdirs = [root / dt for dt in data_types]
        return self.paths.ensure_dirs(*outdirs)

    def make_images_folders(self, machine_name: str, experiment_name: str, image_types: List[str]) -> List[Path]:
        root = self.paths.images_root(machine_name, experiment_name)
        outdirs = [root / it for it in image_types]
        return self.paths.ensure_dirs(*outdirs)
