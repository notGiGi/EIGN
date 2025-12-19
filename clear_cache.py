#!/usr/bin/env python
"""Clear Python cache files (.pyc and __pycache__ directories)"""
import shutil
from pathlib import Path

def clear_cache(root_dir: Path) -> None:
    count_pycache = 0
    count_pyc = 0

    # Remove __pycache__ directories
    for pycache_dir in root_dir.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            count_pycache += 1
            print(f"Removed: {pycache_dir}")
        except Exception as e:
            print(f"Failed to remove {pycache_dir}: {e}")

    # Remove .pyc files
    for pyc_file in root_dir.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            count_pyc += 1
            print(f"Removed: {pyc_file}")
        except Exception as e:
            print(f"Failed to remove {pyc_file}: {e}")

    print(f"\nCleared {count_pycache} __pycache__ directories and {count_pyc} .pyc files")

if __name__ == "__main__":
    root = Path(__file__).parent
    print(f"Clearing cache in: {root}\n")
    clear_cache(root)
