"""Create the configured pipeline directories without reading any source data."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import DataConfigError, load_config, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    logger = None
    try:
        config = load_config(args.config)
        paths = config["paths"]
        directory_keys = ("raw", "interim", "out", "build", "prompts", "tests", "logs")
        for key in (*directory_keys, "dropflow"):
            if not isinstance(paths[key], str) or not paths[key].strip():
                raise DataConfigError(f"paths.{key} must be a non-empty path string")
        logger = setup_logging(Path(__file__).stem, paths["logs"], level=config["logging"]["level"])
        for key in directory_keys:
            Path(paths[key]).mkdir(parents=True, exist_ok=True)
        # Task 00 has no filter stages, so it emits no artificial dropflow rows.
        Path(paths["dropflow"]).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Scaffold is ready.")
    except (DataConfigError, KeyError, TypeError, ValueError, OSError) as exc:
        message = f"Scaffold stopped: {exc}"
        if logger is None:
            print(f"ERROR {message}", flush=True)
        else:
            logger.error(message)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
