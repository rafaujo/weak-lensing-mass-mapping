"""Run only production MCALens C0/R10/P0 in the dedicated mcalens interpreter."""
import argparse
import importlib
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def file_state(path):
    if not path.exists():
        return {"exists": False}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    return {"exists": True, "sha256": digest, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_packaged_driver(paths):
    # The checkout can itself contain a historical run_mcalens_test.py.
    # A bare import by name would prefer it after prepending mcalens_root.
    source = ROOT / "code/pipeline/run_mcalens_test.py"
    name = "wl_packaged_mcalens_test"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    expected = {"BASE": paths.work_root / "test/fair_npz",
                "OUT": paths.work_root / "test/mcalens",
                "RECON_DIR": paths.work_root / "test/mcalens/recon",
                "PK_FILE": paths.prior_file,
                "MASK_FILE": paths.data_root / "WIDE12H_bin2_2arcmin_mask.npy"}
    for key, value in expected.items():
        if Path(getattr(module, key)).resolve() != value.resolve():
            raise ValueError(f"Packaged MCALens {key} escaped the configured paths")
    return module, {"path": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "validated_paths": {key: str(value) for key, value in expected.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["WL_PATHS_CONFIG"] = str(args.paths.absolute())
    sys.path.insert(0, str(ROOT / "code/pipeline"))
    from project_paths import PATHS
    if not args.output.resolve().is_relative_to(PATHS.work_root) or args.output.exists():
        raise ValueError("Worker report must be a new file in this run's work_root")
    if args.paths.resolve().parent != PATHS.work_root:
        raise ValueError("Smoke paths.toml must be inside its dedicated work_root")
    sys.path.insert(0, str(PATHS.mcalens_root))
    report = {"case": [0, 10, 0], "python_executable": sys.executable,
              "python_version": sys.version, "prefix": sys.prefix, "ok": False}
    try:
        historical = PATHS.glimpse_root / "test/mcalens/recon/C0_R10_P0.npz"
        before = file_state(historical)
        mc, report["driver"] = load_packaged_driver(PATHS)
        expected_recon = PATHS.work_root / "test/mcalens/recon/C0_R10_P0.npz"
        if expected_recon.exists():
            raise ValueError("MCALens output already exists; use a new smoke directory")
        report["result"] = mc.run_one((0, 10, 0))
        if Path(report["result"]["recon_file"]).resolve() != expected_recon.resolve() or not expected_recon.is_file():
            raise ValueError("MCALens did not produce its reconstruction inside this run")
        after = file_state(historical)
        report["historical_output_check"] = {"path": str(historical), "before": before, "after": after,
                                              "unchanged": before == after}
        if before != after:
            raise RuntimeError("Historical MCALens output changed during smoke")
        report["mcalens_source"] = importlib.import_module("pycs.astro.wl.mass_mapping").__file__
        report["ok"] = bool(report["result"]["finite"])
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
