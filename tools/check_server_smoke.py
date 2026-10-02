"""Offline checks of smoke isolation, source boundaries, and input validation."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import server_smoke as smoke


class SmokeChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wl smoke checks ")
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.assertTrue(self.folder.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.assertTrue(self.folder.name.startswith("wl smoke checks "))
        self.temp.cleanup()

    def test_prefix_does_not_execute_benchmark(self):
        source = self.folder / "driver.py"
        source.write_text('answer = 42\nmask_full = None\nraise RuntimeError("benchmark started")\n')
        ns = {}
        exec(compile(smoke.prefix_tree(source, "mask_full"), str(source), "exec"), ns)
        self.assertEqual(ns["answer"], 42)
        self.assertNotIn("mask_full", ns)

    def test_ambiguous_or_missing_boundary_fails_closed(self):
        for text in ("x=1\n", "done=set()\ndone=set()\n"):
            source = self.folder / "ambiguous.py"
            source.write_text(text)
            with self.assertRaises(ValueError):
                smoke.prefix_tree(source, "done")

    def test_real_prefixes_stop_before_data_and_resume(self):
        for relative, sentinel, required in (
            ("code/pipeline/run_gaussian_starlet_test.py", "mask_full", "solve"),
            ("code/glimpse/run_glimpse_test_final.py", "done", "write_config"),
        ):
            tree = smoke.prefix_tree(smoke.ROOT / relative, sentinel)
            compile(tree, relative, "exec")
            self.assertTrue(any(isinstance(n, ast.FunctionDef) and n.name == required for n in tree.body))
            self.assertFalse(any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                                 and n.id in ("truths", "fout", "writer") for n in ast.walk(tree)))

    def test_missing_prerequisites_leave_historical_tree_untouched(self):
        history = self.folder / "existing experiment"
        history.mkdir()
        marker = history / "preserve.txt"
        marker.write_text("historical output")
        config = self.folder / "paths.toml"
        values = {"data_root": "missing raw", "work_root": "new outputs",
                  "glimpse_root": "existing experiment", "mcalens_root": "missing mcalens"}
        config.write_text("".join(f"{k}={json.dumps(v)}\n" for k, v in values.items()))
        environments = self.folder / "environments.toml"
        environments.write_text(f"gob5_python={json.dumps([sys.executable])}\n"
                                f"mcalens_python={json.dumps([sys.executable])}\nglimpse_prefix=[]\n")
        env = os.environ.copy()
        env.pop("WL_PATHS_CONFIG", None)
        env.pop("WL_PYTHON", None)
        for _ in range(2):
            result = subprocess.run([sys.executable, "-B", str(smoke.ROOT / "tools/server_smoke.py"),
                                     "--paths", str(config), "--environments", str(environments),
                                     "--preflight-only"], cwd=self.folder,
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        reports = list((self.folder / "new outputs").glob("*/smoke_report.json"))
        self.assertEqual(len(reports), 2)
        for file in reports:
            report = json.loads(file.read_text())
            self.assertFalse(report["numerical_smoke_executed"])
            self.assertEqual(report["status"], "preflight_missing_prerequisites")
            self.assertFalse((file.parent / "test").exists())
        self.assertEqual(list(history.iterdir()), [marker])
        self.assertEqual(marker.read_text(), "historical output")

    def test_archived_references_exist_and_nonmatching_result_is_flagged(self):
        for method in smoke.REFERENCES:
            comp = smoke.reference_comparison(method, dict(nmse=100, pcc=100, std_ratio=100))
            self.assertFalse(comp["within_smoke_tolerance"])
            comp2 = smoke.reference_comparison(method, comp["expected"])
            self.assertTrue(comp2["within_smoke_tolerance"])

    def test_cache_validation_rejects_wrong_truth_noise_and_nonfinite_shear(self):
        import numpy as np
        mask = np.zeros((1424, 176), dtype=bool)
        mask[0, 0] = True
        sampled = np.ones((3, 30, 1), dtype=np.float32)
        labels = np.ones((3, 30, 5))
        square = types.SimpleNamespace(mask_full=mask, sampled=sampled, labels=labels, SIGMA_N=0.02)
        truth = mask.astype(np.float32)
        gamma = np.zeros(truth.shape, dtype=np.complex64)
        cache = self.folder / "C0_R10.npz"
        good = dict(truth=truth, gamma=gamma, omega_m=1.0, sigma_n=0.02, cosmology=0, realization=10)
        np.savez(cache, **good)
        self.assertEqual(smoke.validate_cache(cache, square)[0].shape, truth.shape)
        for key, value in (("truth", truth + 1), ("sigma_n", 0.04), ("realization", 11),
                           ("gamma", gamma + np.nan)):
            np.savez(cache, **dict(good, **{key: value}))
            with self.assertRaises(ValueError):
                smoke.validate_cache(cache, square)


if __name__ == "__main__":
    unittest.main(verbosity=2)
