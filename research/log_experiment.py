#!/usr/bin/env python3
"""
Minimal experiment logger to create a run folder and append an index entry.
No external deps beyond Python stdlib.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Dict

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "research"
EXPERIMENTS_DIR = RESEARCH_DIR / "experiments"
INDEX_FILE = RESEARCH_DIR / "experiments.jsonl"


def _timestamp_now() -> str:
    """Gets the current timestamp as a string.

    Returns:
        The current timestamp in the format YYYYMMDD_HHMMSS.
    """
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _slugify(name: str) -> str:
    """Slugifies a string.

    Args:
        name: The string to slugify.

    Returns:
        The slugified string.
    """
    keep = [c if c.isalnum() or c in ("-", "_") else "-" for c in name.strip().lower()]
    return "".join(keep).strip("-") or "run"


def _git_commit() -> str | None:
    """Gets the current Git commit hash.

    Returns:
        The Git commit hash as a string, or None if it cannot be determined.
    """
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _python_env() -> Dict[str, Any]:
    """Gets information about the Python environment.

    Returns:
        A dictionary with information about the Python environment.
    """
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
    }


def _os_env() -> Dict[str, Any]:
    """Gets information about the OS environment.

    Returns:
        A dictionary with information about the OS environment.
    """
    return {
        "os_name": os.name,
        "env_sample": {k: os.environ.get(k) for k in [
            "USERNAME", "COMPUTERNAME", "NUMBER_OF_PROCESSORS"
        ] if k in os.environ}
    }


def _write_json(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    """Writes a dictionary to a JSON file.

    Args:
        path: The path to the JSON file.
        obj: The dictionary to write.
    """
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _copy_artifact(src: pathlib.Path, dst_dir: pathlib.Path) -> str:
    """Copies an artifact to a destination directory.

    Args:
        src: The path to the source artifact.
        dst_dir: The path to the destination directory.

    Returns:
        The relative path of the copied artifact with respect to the research
        directory.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return str(dst.relative_to(RESEARCH_DIR))


def main() -> None:
    """The main function."""
    ap = argparse.ArgumentParser(description="Log an experiment run")
    ap.add_argument("--name", required=True, help="Short name/slug for the run")
    ap.add_argument("--config", required=True, help="Path to config JSON")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--metrics", help="Metrics JSON string, e.g., '{""fps"":30}'")
    g.add_argument("--metrics-file", help="Path to metrics JSON file")
    ap.add_argument("--notes", help="Optional notes markdown file")
    ap.add_argument("--artifact", action="append", default=[], help="Path to an artifact file (repeatable)")
    ap.add_argument("--tags", help="Comma-separated tags")

    args = ap.parse_args()

    # Load config
    cfg_path = pathlib.Path(args.config).resolve()
    if not cfg_path.exists():
        ap.error(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    # Load metrics
    if args.metrics:
        try:
            metrics = json.loads(args.metrics)
        except json.JSONDecodeError as e:
            ap.error(f"Invalid --metrics JSON: {e}")
    else:
        m_path = pathlib.Path(args.metrics_file).resolve()
        if not m_path.exists():
            ap.error(f"Metrics file not found: {m_path}")
        with m_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

    # Prepare run folder
    ts = _timestamp_now()
    slug = _slugify(args.name)
    run_dir = EXPERIMENTS_DIR / f"{ts}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir = run_dir / "artifacts"

    # Save files
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metrics.json", metrics)

    # Notes
    if args.notes:
        notes_src = pathlib.Path(args.notes).resolve()
        if not notes_src.exists():
            ap.error(f"Notes file not found: {notes_src}")
        shutil.copy2(notes_src, run_dir / "notes.md")
    else:
        (run_dir / "notes.md").write_text("# Notes\n\n- fill in rationale and observations\n", encoding="utf-8")

    # Environment snapshot
    env_snapshot = {
        "git_commit": _git_commit(),
        "python": _python_env(),
        "os": _os_env(),
    }
    _write_json(run_dir / "env.json", env_snapshot)

    # Artifacts
    rel_artifacts = []
    for a in args.artifact:
        p = pathlib.Path(a).resolve()
        if p.exists():
            rel_artifacts.append(_copy_artifact(p, artifacts_dir))
        else:
            print(f"[warn] artifact not found, skipping: {p}")

    # Build index entry
    entry = {
        "timestamp": ts,
        "name": slug,
        "run_dir": str(run_dir.relative_to(RESEARCH_DIR)),
        "tags": [t.strip() for t in (args.tags or "").split(",") if t.strip()],
        "metrics": metrics,
        "hash": hashlib.sha1(json.dumps(metrics, sort_keys=True).encode()).hexdigest()[:10],
        "git": env_snapshot.get("git_commit"),
    }

    with INDEX_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"Saved run at: {run_dir}")
    print(f"Indexed in: {INDEX_FILE}")


if __name__ == "__main__":
    main()
