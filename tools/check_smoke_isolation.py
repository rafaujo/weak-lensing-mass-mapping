"""Regression tests for escaped MCALens output and unwanted GS cube allocation."""
import ast
import json
from pathlib import Path
import tempfile
import types
import sys
import unittest
from unittest.mock import patch

import mcalens_smoke_worker as worker
import server_smoke as smoke


class IsolationChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wl isolation checks ")
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.assertTrue(self.folder.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.assertTrue(self.folder.name.startswith("wl isolation checks "))
        self.temp.cleanup()

    def test_gs_loader_does_not_allocate_any_full_benchmark_cube(self):
        source = self.folder / "gs.py"
        source.write_text('''REC_FILE="rec.npy"
GAUSS_FILE="gauss.npy"
SPARSE_FILE="sparse.npy"
def open_output_map(path):
    raise RuntimeError("Full benchmark allocation reached")
rec_map=open_output_map(REC_FILE)
gauss_map=open_output_map(GAUSS_FILE)
sparse_map=open_output_map(SPARSE_FILE)
def solve():
    return "production solver definition retained"
mask_full=None
raise RuntimeError("Full benchmark loop reached")
''')
        tree = smoke.prefix_tree(source, "mask_full", exclude_allocations=True)
        values = {}
        exec(compile(tree, str(source), "exec"), values)
        self.assertEqual(values["solve"](), "production solver definition retained")
        self.assertTrue(all(name not in values for name in ("rec_map", "gauss_map", "sparse_map")))

    def test_actual_gs_solver_definitions_are_unchanged_by_loader(self):
        source = smoke.ROOT / "code/pipeline/run_gaussian_starlet_test.py"
        original = ast.parse(source.read_bytes())
        selected = smoke.prefix_tree(source, "mask_full", exclude_allocations=True)
        definitions = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        before, after = definitions(original), definitions(selected)
        self.assertTrue(all(before[name] == dump for name, dump in after.items()))
        self.assertIn("solve", after)
        self.assertIn("FourierStarlet", after)

    def test_changed_gs_allocation_fails_closed(self):
        source = self.folder / "gs.py"
        source.write_text('rec_map=unexpected_call()\nmask_full=None\n')
        with self.assertRaises(ValueError):
            smoke.prefix_tree(source, "mask_full", exclude_allocations=True)

    def test_packaged_driver_wrong_output_root_rejected_before_run(self):
        repo = self.folder / "repo"
        code = repo / "code/pipeline"
        code.mkdir(parents=True)
        historical = self.folder / "historical.npz"
        historical.write_bytes(b"preserve")
        (code / "run_mcalens_test.py").write_text('from pathlib import Path\n'
            f'BASE=Path({str(self.folder / "wrong").__repr__()})\n'
            f'def run_one(case):\n    Path({str(historical).__repr__()}).write_bytes(b"modified")\n')
        paths = types.SimpleNamespace(work_root=self.folder / "run", prior_file=self.folder / "prior.npy",
                                      data_root=self.folder / "raw")
        with patch.object(worker, "ROOT", repo), self.assertRaises(ValueError):
            worker.load_packaged_driver(paths)
        self.assertEqual(historical.read_bytes(), b"preserve")

    def test_parent_rejects_success_report_pointing_to_historical_result(self):
        run = self.folder / "run"
        run.mkdir()
        cfg = run / "paths.toml"
        cfg.write_text('data_root="raw"\nwork_root="run"\nglimpse_root="glimpse"\nmcalens_root="mc"\n')
        (run / "mcalens_worker.json").write_text(json.dumps({"ok": True,
            "result": {"recon_file": str(self.folder / "historical.npz")}}))
        with patch.object(smoke.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)):
            with self.assertRaisesRegex(RuntimeError, "outside this smoke run"):
                smoke.run_mcalens_process({"mcalens_python": ["synthetic"]}, cfg, run)

    def test_mcalens_only_does_not_run_core_solvers_or_glimpse(self):
        import numpy as np
        run = self.folder / "run"
        run.mkdir()
        old_cache = self.folder / "historical/test/fair_npz"
        old_cache.mkdir(parents=True)
        (old_cache / "C0_R10.npz").write_bytes(b"synthetic cache fixture")
        truth = np.zeros((1424, 176), dtype=np.float32)
        gamma = truth.astype(np.complex64)
        mask = np.ones(truth.shape, dtype=bool)
        rec = run / "mc_fixture.npz"
        np.savez(rec, mcalens=truth[:176])
        square = types.SimpleNamespace(DEVICE="cpu", N=176, mask_full=mask,
                                       metrics_ms=lambda *args: (0.0, 0.0, 0.0))
        torch = types.ModuleType("torch")
        torch.__version__ = "fixture"
        torch.version = types.SimpleNamespace(cuda=None)
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.float32, torch.complex64 = np.float32, np.complex64
        torch.as_tensor = lambda value, **kwargs: value
        args = types.SimpleNamespace(generate_input=False, mcalens_only=True,
                                     with_glimpse=False, environments_config={})
        paths = types.SimpleNamespace(glimpse_root=self.folder / "historical")
        report = {}
        with patch.dict(sys.modules, {"torch": torch}), \
             patch.object(smoke.importlib, "import_module", return_value=square), \
             patch.object(smoke, "validate_cache", return_value=(truth, gamma, 0.3)), \
             patch.object(smoke, "load_prefix", side_effect=AssertionError("GS/GLIMPSE should not load")), \
             patch.object(smoke, "run_mcalens_process", return_value={"result": {"recon_file": str(rec)}}):
            smoke.numerical_run(args, paths, run, report)
        self.assertEqual(set(report["methods"]), {"MCALENS"})
        self.assertTrue(report["numerical_smoke_executed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
