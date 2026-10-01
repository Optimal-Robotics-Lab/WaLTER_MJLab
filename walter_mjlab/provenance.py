"""Record the installed code, model, and dependencies used for an experiment."""

import hashlib
import importlib.metadata
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _git_output(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def collect_provenance(model: Any) -> dict[str, Any]:
    """Hash current source and the compiled model; missing Git metadata stays explicit."""
    import mujoco

    root = Path(__file__).resolve().parents[1]
    dependencies = {}
    for name in (
        "walter-mjlab",
        "mjlab",
        "mujoco",
        "mujoco-warp",
        "torch",
        "rsl-rl-lib",
        "WaLTER_Modeling",
        "pygame",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "model.mjb"
        # MuJoCo's compiled export is missing from its static type stubs.
        mujoco.mj_saveModel(model, str(model_path))  # pyright: ignore[reportAttributeAccessIssue]
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    source_hashes = {
        str(file.relative_to(root)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted((root / "walter_mjlab").rglob("*.py"))
    }
    lock_path = root / "uv.lock"
    return {
        "source_files_sha256": source_hashes,
        "dependency_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest()
        if lock_path.exists()
        else None,
        "dependencies": dependencies,
        "source_revision": _git_output(root, "rev-parse", "HEAD"),
        "source_status": _git_output(root, "status", "--porcelain"),
        "modeling_gitlink": _git_output(root, "ls-files", "--stage", "modeling"),
        "modeling_revision": _git_output(root / "modeling", "rev-parse", "HEAD"),
        "modeling_status": _git_output(root / "modeling", "status", "--porcelain"),
        "generated_model_sha256": model_hash,
    }
