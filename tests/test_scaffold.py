"""Integration checks for the config-driven, offline scaffold command."""

import os
from pathlib import Path
import subprocess
import sys

import yaml

from data.build.common import load_config


def test_scaffold_uses_config_and_is_idempotent(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/data.yaml")
    config["paths"] = {key: str(tmp_path / value) for key, value in config["paths"].items()}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    command = [sys.executable, "-B", str(root / "data/build/00_scaffold.py"), "--config", str(config_path)]
    first = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert first.stdout == "INFO Scaffold is ready.\n"
    log = Path(config["paths"]["logs"]) / "00_scaffold.log"
    log_bytes = log.read_bytes()
    second = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert second.stdout == first.stdout
    assert log.read_bytes() == log_bytes
    for key, value in config["paths"].items():
        if key != "dropflow":
            assert Path(value).is_dir()
    assert not Path(config["paths"]["dropflow"]).exists()


def test_scaffold_stops_with_diagnostic_on_missing_paths(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths: {}\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-B", str(root / "data/build/00_scaffold.py"), "--config", str(config_path)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "ERROR Scaffold stopped:" in result.stdout
    assert "raw" in result.stdout


def test_make_runs_scaffold_in_current_environment() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["make", "-n", "all", f"PYTHON={sys.executable}"],
        cwd=root, capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert f"{sys.executable} -B data/build/00_scaffold.py --config" in result.stdout
