"""Static checks only. For actual per-environment imports use server_smoke.py --preflight-only."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
from project_paths import load_paths


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(config=None):
    paths = load_paths(config)
    raw = json.loads((ROOT / "provenance/raw_data_checksums.json").read_text(encoding="utf-8"))
    inputs = []
    for entry in raw:
        path = paths.data_root / PurePosixPath(entry["path"]).name
        exists = path.is_file()
        inputs.append({"path": str(path), "exists": exists,
                       "sha256_matches": file_sha(path) == entry["sha256"] if exists else False})
    inputs.append({"path": str(paths.prior_file), "exists": paths.prior_file.is_file(),
                   "sha256_matches": paths.prior_file.is_file() and file_sha(paths.prior_file) == next(
                       line.split()[0] for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines()
                       if line.endswith("config/priors/fair_validation_pk_mean.npy"))})
    modules = {}
    for module, package in [("numpy", "numpy"), ("torch", "torch"), ("scipy", "scipy"),
                            ("pandas", "pandas"), ("astropy", "astropy"),
                            ("pywt", "PyWavelets"), ("matplotlib", "matplotlib")]:
        found = importlib.util.find_spec(module) is not None
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        modules[module] = {"discoverable": found, "version": version}
    versions = json.loads((ROOT / "provenance/software_versions.json").read_text())
    recorded_head = versions["mcalens"]["head"]
    expected_commit = (recorded_head["text"] if isinstance(recorded_head, dict) else recorded_head).strip()
    mcalens = {"checkout_present": (paths.mcalens_root / "pycs/astro/wl/mass_mapping.py").is_file(),
               "expected_commit": expected_commit, "observed_commit": None}
    if mcalens["checkout_present"]:
        try:
            result = subprocess.run(["git", "-C", str(paths.mcalens_root), "rev-parse", "HEAD"],
                                    text=True, capture_output=True, timeout=15)
            if result.returncode == 0:
                mcalens["observed_commit"] = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    mcalens["commit_matches"] = mcalens["observed_commit"] == expected_commit
    expected_patch = (ROOT / "provenance/mcalens/local_source_changes.patch").read_text(encoding="utf-8")
    rows = expected_patch.splitlines(keepends=True)
    start = next(i for i, line in enumerate(rows) if line.startswith("@@ "))
    patched = "".join(line[1:] for line in rows[start + 1:] if line.startswith(("+", " ")))
    starlet = paths.mcalens_root / "pycs/sparsity/sparse2d/starlet.py"
    mcalens["local_starlet_patch_matches"] = starlet.is_file() and starlet.read_text(encoding="utf-8") == patched
    data_ok = all(row["sha256_matches"] for row in inputs)
    modules_ok = all(row["discoverable"] for row in modules.values())
    return {"python": sys.version, "paths": paths.as_dict(), "inputs": inputs, "modules": modules,
            "mcalens": mcalens, "glimpse_executable_present": paths.glimpse_executable.is_file(),
            "core_prerequisites_present": data_ok and modules_ok,
            "numerical_smoke_executed": False,
            "scope": "File hashes, module discovery and checkout metadata only; no import/solver/device validation or numerical execution."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = preflight(args.paths)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
