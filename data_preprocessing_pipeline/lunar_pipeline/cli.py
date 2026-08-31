from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lunar_pipeline.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="lunar-prep",
        description="Ingest Chandrayaan-2 PDS4 products and emit normalized GeoTIFF tiles + metadata catalog.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run the full standardization pipeline")
    run.add_argument("input_dir", type=Path, help="Folder of PDS4 .xml/.img pairs, GeoTIFFs, or .cub files")
    run.add_argument("--out", type=Path, required=True, help="Output directory")
    run.add_argument("--config", type=Path, default=None, help="YAML config (defaults to config/default.yaml)")

    args = p.parse_args(argv)
    if args.cmd == "run":
        records = run_pipeline(args.input_dir, args.out, args.config)
        print(f"Wrote {len(records)} tiles -> {args.out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
