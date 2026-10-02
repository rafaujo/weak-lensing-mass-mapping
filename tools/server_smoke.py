"""Run exactly C0/R10/P0 using the unchanged production solver functions.

No benchmark driver is launched. GS/GLIMPSE modules have top-level benchmark
loops: only their source prefix, ending before an explicit sentinel, is loaded.
Source integrity is verified before imports. All outputs go to a new directory.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import types
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/pipeline"))
from project_paths import load_paths
from smoke_preflight import preflight, file_sha
from verify_scientific_invariants import verify
from execution_environments import load_environments, python_command, process_environment

CASE = (0, 10, 0)
REFERENCE_TOLERANCE = 1e-4  # smoke diagnostic only; never used inside a solver
REFERENCES = {
    "KS": "results/test/square/square_ours_metrics.csv",
    "STARLET_MASK": "results/test/square/square_ours_metrics.csv",
    "GAUSSIAN_STARLET": "results/test/starlet_gaussian_prior/gaussian_starlet_test_metrics.csv",
    "MCALENS": "results/test/mcalens/mcalens_test_metrics.csv",
    "GLIMPSE": "results/test/glimpse_test_metrics.csv",
}


def prefix_tree(path, sentinel, exclude_allocations=False):
    """Stop before exactly one top-level assignment; never include later loops."""
    tree = ast.parse(path.read_bytes(), filename=str(path))
    indices = [i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == sentinel for t in node.targets)]
    if len(indices) != 1:
        raise ValueError(f"Expected one boundary {sentinel!r} in {path}, got {indices}")
    tree.body = tree.body[:indices[0]]
    if exclude_allocations:
        # Production GS allocates three full-benchmark cubes before mask_full.
        # The smoke calls solve() directly and needs none of these output buffers.
        expected = {"rec_map": "REC_FILE", "gauss_map": "GAUSS_FILE", "sparse_map": "SPARSE_FILE"}
        found = set()
        body = []
        for node in tree.body:
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else []
            matching = set(targets) & set(expected)
            if matching:
                name = next(iter(matching))
                reference = ast.parse(f"{name} = open_output_map({expected[name]})").body[0]
                if ast.dump(node, include_attributes=False) != ast.dump(reference, include_attributes=False):
                    raise ValueError(f"Unexpected GS allocation: {name}")
                found.add(name)
            else:
                body.append(node)
        if found != set(expected):
            raise ValueError("Expected exactly the three production GS output allocations")
        tree.body = body
    return tree


def load_prefix(relative, sentinel):
    path = ROOT / relative
    mod = types.ModuleType("smoke_" + path.stem)
    mod.__file__ = str(path)
    tree = prefix_tree(path, sentinel, exclude_allocations=relative == "code/pipeline/run_gaussian_starlet_test.py")
    exec(compile(tree, str(path), "exec"), mod.__dict__)
    return mod


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def reference_comparison(method, values):
    with (ROOT / REFERENCES[method]).open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["method"] == method
                and tuple(int(r[k]) for k in ("c", "r", "p")) == CASE]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one archived reference for {method}")
    expected = {k: float(rows[0][k]) for k in ("nmse", "pcc", "std_ratio")}
    delta = {k: values[k] - expected[k] for k in expected}
    return {"source": REFERENCES[method], "expected": expected, "delta": delta,
            "absolute_tolerance": REFERENCE_TOLERANCE,
            "within_smoke_tolerance": all(abs(v) <= REFERENCE_TOLERANCE for v in delta.values())}


def validate_cache(path, square):
    import numpy as np
    with np.load(path) as data:
        truth = data["truth"].copy()
        gamma = data["gamma"].copy()
        omega = float(data["omega_m"])
        sigma = float(data["sigma_n"])
        for key, expected in (("cosmology", CASE[0]), ("realization", CASE[1])):
            if key in data and int(data[key]) != expected:
                raise ValueError(f"Wrong {key} in cache")
    expected_truth = np.zeros(square.mask_full.shape, dtype=np.float32)
    expected_truth[square.mask_full] = square.sampled[CASE[0], CASE[1]].astype(np.float32)
    if truth.shape != (1424, 176) or gamma.shape != truth.shape:
        raise ValueError("Cache has unexpected full-survey shape")
    if not np.array_equal(truth, expected_truth) or not np.isfinite(gamma).all():
        raise ValueError("Cache truth differs from FAIR inputs or shear contains nonfinite values")
    if not np.isclose(sigma, square.SIGMA_N, rtol=0, atol=1e-12):
        raise ValueError("Cache has different shape-noise sigma")
    if not np.isclose(omega, float(square.labels[CASE[0], CASE[1], 0]), rtol=0, atol=1e-12):
        raise ValueError("Cache cosmology differs from FAIR labels")
    return truth, gamma, omega


def probe_environments(settings, runtime_config, run_dir, with_glimpse=False):
    paths = load_paths(runtime_config)
    reports = {}
    for role in ("gob5", "mcalens"):
        output = run_dir / f"imports_{role}.json"
        command = [*python_command(settings, role), "-B", str(ROOT / "tools/environment_probe.py"),
                   "--kind", role, "--paths", str(runtime_config), "--output", str(output)]
        try:
            with (run_dir / f"imports_{role}.log").open("w", encoding="utf-8") as log:
                proc = subprocess.run(command, env=process_environment(paths, role), stdout=log,
                                      stderr=subprocess.STDOUT, timeout=120)
            reports[role] = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {
                "ok": False, "error": f"No import report; inspect imports_{role}.log"}
            reports[role]["ok"] = reports[role]["ok"] and proc.returncode == 0
            reports[role]["returncode"] = proc.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            reports[role] = {"ok": False, "error": str(exc)}
        reports[role]["command"] = command
        print(f"{role} imports: {'OK' if reports[role]['ok'] else 'FAILED'}", flush=True)
        for name, details in reports[role].get("imports", {}).items():
            if not details["ok"]:
                print(f"  {name}: {details.get('error', 'wrong source/classes')}", flush=True)
        if "error" in reports[role]:
            print(" ", reports[role]["error"], flush=True)
    if with_glimpse:
        command = [*settings["glimpse_prefix"], "ldd", str(paths.glimpse_executable)]
        try:
            with (run_dir / "imports_glimpse.log").open("w", encoding="utf-8") as log:
                proc = subprocess.run(command, env=process_environment(paths, "glimpse"), stdout=log,
                                      stderr=subprocess.STDOUT, timeout=60)
            diagnostic = (run_dir / "imports_glimpse.log").read_text(encoding="utf-8", errors="replace")
            reports["glimpse"] = {"ok": proc.returncode == 0 and "not found" not in diagnostic,
                                  "returncode": proc.returncode,
                                  "scope": "Native executable dynamic-library resolution via ldd; not a reconstruction"}
        except (OSError, subprocess.TimeoutExpired) as exc:
            reports["glimpse"] = {"ok": False, "error": str(exc)}
        reports["glimpse"]["command"] = command
    return reports


def run_mcalens_process(settings, runtime_config, run_dir):
    output = run_dir / "mcalens_worker.json"
    command = [*python_command(settings, "mcalens"), "-B", "-u",
               str(ROOT / "tools/mcalens_smoke_worker.py"),
               "--paths", str(runtime_config), "--output", str(output)]
    with (run_dir / "mcalens.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, env=process_environment(load_paths(runtime_config), "mcalens"),
                              stdout=log, stderr=subprocess.STDOUT)
    details = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
    if proc.returncode != 0 or not details.get("ok"):
        raise RuntimeError(f"MCALens child failed: {details.get('error', 'see mcalens.log')}; {run_dir}")
    expected = (run_dir / "test/mcalens/recon/C0_R10_P0.npz").resolve()
    source = ROOT / "code/pipeline/run_mcalens_test.py"
    if Path(details["result"]["recon_file"]).resolve() != expected or not expected.is_file():
        raise RuntimeError("MCALens returned a reconstruction outside this smoke run")
    if (Path(details.get("driver", {}).get("path", "")).resolve() != source.resolve()
            or details["driver"]["sha256"] != file_sha(source)):
        raise RuntimeError("MCALens worker did not attest the packaged scientific driver")
    if not details.get("historical_output_check", {}).get("unchanged"):
        raise RuntimeError("Missing/failed historical-output integrity check")
    details["command"] = command
    return details


def numerical_run(args, paths, run_dir, report):
    import numpy as np
    import torch
    square = importlib.import_module("run_square_starlet_comparison")
    report["device"] = str(square.DEVICE)
    report["torch_version"] = str(torch.__version__)
    report["cuda_version"] = torch.version.cuda
    report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    cache_dir = run_dir / "test/fair_npz"
    cache_dir.mkdir(parents=True)
    cache = cache_dir / "C0_R10.npz"
    historical = paths.glimpse_root / "test/fair_npz/C0_R10.npz"
    if args.generate_input:
        print("Generating the full C0/R10 observation with the original seed=10 function.", flush=True)
        square.load_or_generate_case(CASE[0], CASE[1], cache_dir)
        origin = "regenerated with original function; CPU/CUDA RNG streams can differ"
    else:
        if not historical.is_file():
            raise FileNotFoundError(f"Historical cache missing: {historical}. Use --generate-input explicitly to generate it.")
        shutil.copyfile(historical, cache)
        if file_sha(cache) != file_sha(historical):
            raise RuntimeError("Historical cache copy checksum mismatch")
        origin = str(historical)
    truth_full, gamma_full, omega = validate_cache(cache, square)
    report["input"] = {"origin": origin, "sha256": file_sha(cache),
                       "historical_gamma_hash_independently_archived": False,
                       "truth_noise_scale_and_cosmology_validated": True}
    truth, gamma = truth_full[:square.N].copy(), gamma_full[:square.N].copy()
    mask = square.mask_full[:square.N].copy()
    y = torch.as_tensor((gamma * mask)[None], dtype=torch.complex64, device=square.DEVICE)
    m = torch.as_tensor(mask[None], dtype=torch.float32, device=square.DEVICE)
    report["methods"] = {}
    maps_dir = run_dir / "maps"
    maps_dir.mkdir()

    def record(method, rec, started):
        rec = np.asarray(rec)
        if rec.shape != truth.shape or not np.isfinite(rec).all():
            raise ValueError(f"Invalid reconstruction for {method}")
        vals = dict(zip(("nmse", "pcc", "std_ratio"), square.metrics_ms(rec, truth, mask)))
        if not all(np.isfinite(v) for v in vals.values()):
            raise ValueError(f"Nonfinite metrics for {method}")
        np.save(maps_dir / (method + ".npy"), rec)
        report["methods"][method] = {"metrics": vals, "runtime_s": time.perf_counter() - started,
                                    "shape": list(rec.shape), "finite": True,
                                    "comparison": reference_comparison(method, vals)}
        write_json(run_dir / "smoke_report.json", report)
        print(method, vals, flush=True)

    if not args.mcalens_only:
        print("KS, then Starlet with the production 6000 iterations...", flush=True)
        report["numerical_smoke_executed"] = True
        started = time.perf_counter()
        with torch.no_grad():
            ks_rec = square.ks_reconstruct(y, square.G15)
            x0 = square.ks_reconstruct(y, square.G1)
        record("KS", ks_rec[0].cpu().numpy(), started)
        started = time.perf_counter()
        starlet = square.starlet_pdhg(y, m, x0)
        record("STARLET_MASK", starlet[0].cpu().numpy(), started)

        print("Gaussian+Starlet with the production 6000 iterations...", flush=True)
        gs = load_prefix("code/pipeline/run_gaussian_starlet_test.py", "mask_full")
        op = gs.WeakLensingOperator(gs.N, gs.N, gs.DEVICE)
        ks = gs.KaiserSquires(op)
        star = gs.FourierStarlet(gs.N, gs.N, gs.J, gs.DEVICE)
        pk_np = gs.make_pk_map(np.load(gs.PK_FILE).astype(np.float64), gs.N, gs.N)
        pk = torch.as_tensor(pk_np, dtype=torch.float32, device=gs.DEVICE).unsqueeze(0)
        alpha = torch.as_tensor(gs.ALPHA, dtype=torch.float32, device=gs.DEVICE).view(1, gs.J, 1, 1)
        x0 = gs.ks_reconstruct_batch(ks, y, smooth_sigma=1.0)
        if torch.is_complex(x0):
            x0 = x0.real
        started = time.perf_counter()
        rec, xg, xs, rel = gs.solve(op=op, star=star, y=y, mask=m, pk=pk, alpha=alpha,
                                   x0=x0.to(dtype=torch.float32))
        record("GAUSSIAN_STARLET", rec[0].cpu().numpy(), started)
        report["gs_last_relative_change"] = float(rel[0].cpu())

    print("MCALens with the production Nsigma=5, 100 iterations, four scales...", flush=True)
    started = time.perf_counter()
    report["numerical_smoke_executed"] = True
    mc_worker = run_mcalens_process(args.environments_config, run_dir / "paths.toml", run_dir)
    mc_result = mc_worker["result"]
    with np.load(mc_result["recon_file"]) as data:
        record("MCALENS", data["mcalens"], started)
    report["mcalens_worker"] = mc_worker

    if args.with_glimpse:
        print("GLIMPSE: this single case took about 18 minutes in the archived run.", flush=True)
        gl = load_prefix("code/glimpse/run_glimpse_test_final.py", "done")
        cat, cfg, output = gl.CAT_DIR / "C0_R10_P0.fits", gl.CFG_DIR / "C0.ini", gl.REC_DIR / "C0_R10_P0.fits"
        gl.write_catalog(cat, gamma, mask)
        gl.write_config(cfg, omega)
        env = process_environment(paths, "glimpse")
        env["GSL_RNG_SEED"] = str(CASE[0]*10000 + CASE[1]*100 + CASE[2])
        command = [*args.environments_config["glimpse_prefix"], str(gl.GLIMPSE),
                   "--config", str(cfg), "--data", str(cat), "--output", str(output)]
        started = time.perf_counter()
        with (run_dir / "glimpse.log").open("w", encoding="utf-8") as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        record("GLIMPSE", gl.fits.getdata(output).astype(np.float64), started)
        report["glimpse_command"] = command
        report["glimpse_binary_sha256"] = file_sha(gl.GLIMPSE)
        report["glimpse_rng_seed"] = env["GSL_RNG_SEED"]
    else:
        report["glimpse"] = "not requested; add --with-glimpse to test the native binary"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=ROOT / "config_paths.server.toml")
    parser.add_argument("--environments", type=Path, default=ROOT / "config_environments.server.toml")
    parser.add_argument("--preflight-only", action="store_true", help="Check files and actual imports in both environments; no reconstruction")
    methods = parser.add_mutually_exclusive_group()
    methods.add_argument("--with-glimpse", action="store_true", help="Also run the slower native GLIMPSE solver")
    methods.add_argument("--mcalens-only", action="store_true", help="Recheck only MCALens, with input/source/output integrity checks")
    parser.add_argument("--generate-input", action="store_true", help="Generate full-survey C0/R10 shear instead of copying the historical observation")
    args = parser.parse_args()
    args.environments_config = load_environments(args.environments)
    desired_python = args.environments_config["gob5_python"][0]
    if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(os.path.abspath(desired_python)):
        # The shell's active environment must not silently choose the core solver environment.
        return subprocess.run([desired_python, "-B", "-u", str(Path(__file__).absolute()), *sys.argv[1:]]).returncode
    paths = load_paths(args.paths)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = paths.work_root / tag
    run_dir.mkdir(parents=True, exist_ok=False)
    # Importing the production modules can create directories; redirect first.
    runtime_config = run_dir / "paths.toml"
    config = {k: str(getattr(paths, k)) for k in ("data_root", "work_root", "glimpse_root", "mcalens_root")}
    config["work_root"] = str(run_dir)
    runtime_config.write_text("".join(f"{k} = {json.dumps(v)}\n" for k, v in config.items()), encoding="utf-8")
    os.environ["WL_PATHS_CONFIG"] = str(runtime_config)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    import project_paths
    project_paths.PATHS = load_paths(runtime_config)
    report = {"status": "running", "case": list(CASE), "run_directory": str(run_dir),
              "numerical_smoke_executed": False,
              "core_python_executable": sys.executable, "execution_environments": args.environments_config,
              "requested_methods": (["MCALENS"] if args.mcalens_only else
                                    ["KS", "STARLET_MASK", "GAUSSIAN_STARLET", "MCALENS"] +
                                    (["GLIMPSE"] if args.with_glimpse else [])),
              "scope": "One-patch execution and diagnostic comparison, not a full benchmark reproduction."}
    print("Outputs:", run_dir, flush=True)
    code = 1
    try:
        guards = verify()
        if guards["failed"]:
            raise RuntimeError("Scientific source/invariant verification failed")
        report["invariants"] = {k: v for k, v in guards.items() if k not in ("scripts", "unchanged_originals")}
        report["preflight"] = preflight(runtime_config)
        report["environment_imports"] = probe_environments(
            args.environments_config, runtime_config, run_dir, args.with_glimpse)
        pre = report["preflight"]
        ready = (pre["core_prerequisites_present"] and pre["mcalens"]["checkout_present"]
                 and pre["mcalens"]["commit_matches"] and pre["mcalens"]["local_starlet_patch_matches"])
        if args.with_glimpse:
            ready = ready and pre["glimpse_executable_present"]
        ready = ready and all(item["ok"] for item in report["environment_imports"].values())
        if args.preflight_only:
            report["status"] = "preflight_ready" if ready else "preflight_missing_prerequisites"
            code = 0 if ready else 1
        elif not ready:
            raise RuntimeError("Preflight failed. Inspect smoke_report.json; no scientific environment was installed/modified.")
        else:
            numerical_run(args, paths, run_dir, report)
            close = all(r["comparison"]["within_smoke_tolerance"] for r in report["methods"].values())
            report["status"] = "passed" if close else "completed_needs_numerical_review"
            code = 0 if close else 2
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        write_json(run_dir / "smoke_report.json", report)
        print("Status:", report["status"], flush=True)
        print("Report:", run_dir / "smoke_report.json", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
