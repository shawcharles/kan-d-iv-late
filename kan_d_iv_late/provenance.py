from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run_git_command(args, cwd):
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _git_metadata(path):
    path = Path(path).resolve()
    if path.is_file():
        path = path.parent
    root = _run_git_command(["rev-parse", "--show-toplevel"], path)
    if root is None:
        return {"path": str(path), "available": False}

    root_path = Path(root)
    status = _run_git_command(["status", "--porcelain"], root_path)
    return {
        "path": str(root_path),
        "available": True,
        "commit": _run_git_command(["rev-parse", "HEAD"], root_path),
        "branch": _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], root_path),
        "dirty": bool(status),
    }


def _package_version(distribution_name):
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _efficient_kan_source_path(repo_root):
    module = sys.modules.get("efficient_kan")
    module_file = getattr(module, "__file__", None)
    if module_file:
        return Path(module_file).resolve()

    try:
        spec = importlib.util.find_spec("efficient_kan")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        return Path(spec.origin).resolve()

    sibling_checkout = Path(repo_root).resolve().parent / "efficient-kan"
    if sibling_checkout.exists():
        return sibling_checkout
    return None


def collect_run_provenance(project_root=None):
    """Return manifest-safe provenance for generated numerical artifacts."""
    package_dir = Path(__file__).resolve().parent
    repo_root = Path(project_root).resolve() if project_root is not None else package_dir.parent
    efficient_kan_path = _efficient_kan_source_path(repo_root)

    repositories = {
        "kan-d-iv-late": _git_metadata(repo_root),
    }
    if efficient_kan_path is not None:
        repositories["efficient-kan"] = _git_metadata(efficient_kan_path)
    else:
        repositories["efficient-kan"] = {"available": False}

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "cwd": str(Path.cwd()),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {
            "efficient-kan": _package_version("efficient-kan"),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit-learn": _package_version("scikit-learn"),
            "torch": _package_version("torch"),
        },
        "repositories": repositories,
    }
