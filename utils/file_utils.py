"""
File-system utility functions for the Agentic AI project.

Provides helpers for directory validation, recursive file scanning,
hidden-file detection, and safe file copying used across agents.
"""

import shutil
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Hidden-file helpers ─────────────────────────────────────────────────────

def is_hidden_file(file_path: Path) -> bool:
    """
    Check whether a file itself is hidden.

    A file is considered hidden if its **name** starts with a dot
    (cross-platform convention).

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        ``True`` if the file name starts with ``'.'``.
    """
    return file_path.name.startswith(".")


def is_hidden_path(file_path: Path, root: Path) -> bool:
    """
    Check whether *any* component of the path (relative to *root*)
    is hidden (i.e. starts with a dot).

    This catches files inside hidden directories such as
    ``root/.hidden_dir/image.jpg``.

    Args:
        file_path: Absolute path to the file.
        root:      The root directory against which the relative
                   path is computed.

    Returns:
        ``True`` if any component of the relative path is hidden.
    """
    try:
        relative = file_path.relative_to(root)
        return any(part.startswith(".") for part in relative.parts)
    except ValueError:
        # file_path is not relative to root – fall back to filename check
        return is_hidden_file(file_path)


# ── Extension helper ────────────────────────────────────────────────────────

def get_file_extension(file_path: Path) -> str:
    """
    Return the lowercase file extension **without** the leading dot.

    Examples::

        get_file_extension(Path("image.JPG"))  # → "jpg"
        get_file_extension(Path("readme"))     # → ""

    Args:
        file_path: Path to the file.

    Returns:
        Lowercase extension string (e.g. ``"jpg"``, ``"png"``).
    """
    return file_path.suffix.lstrip(".").lower()


# ── Directory helpers ───────────────────────────────────────────────────────

def ensure_directory_exists(directory: Path) -> None:
    """
    Create *directory* (and all parents) if it does not already exist.

    Args:
        directory: Path to create.
    """
    directory.mkdir(parents=True, exist_ok=True)


def clear_directory(directory: Path) -> None:
    """
    Remove all contents of *directory* if it exists, then recreate it.

    Used to guarantee a fresh output for repeatable pipeline runs.

    Args:
        directory: Path to clear and recreate.
    """
    if directory.exists():
        shutil.rmtree(directory)
        logger.info(f"Cleared existing directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)


# ── Recursive file scanning ────────────────────────────────────────────────

def get_all_files_recursive(directory: Path) -> list[Path]:
    """
    Return a sorted list of every file inside *directory* (recursive).

    Does **not** filter hidden files – that decision is left to the
    calling agent so that hidden files can be counted/logged.

    Args:
        directory: Root directory to scan.

    Returns:
        Sorted list of :class:`~pathlib.Path` objects for every file
        found under *directory*.  Empty list if *directory* does not
        exist.
    """
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    return sorted(f for f in directory.rglob("*") if f.is_file())


# ── File copying ────────────────────────────────────────────────────────────

def copy_file(source: Path, destination: Path) -> bool:
    """
    Copy *source* to *destination*, creating parent directories as needed.

    Uses :func:`shutil.copy2` to preserve file metadata (timestamps,
    permissions).

    Args:
        source:      Path to the source file.
        destination: Path to the destination file.

    Returns:
        ``True`` if the copy succeeded, ``False`` otherwise.
    """
    try:
        ensure_directory_exists(destination.parent)
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        logger.error(f"Failed to copy {source} → {destination}: {e}")
        return False
