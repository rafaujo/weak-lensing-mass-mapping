"""Regression tests for the v4 wrong-interpreter/modopt failure."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import environment_probe
import execution_environments as environments
import server_smoke


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wl environment checks ")
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.assertTrue(self.folder.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.assertTrue(self.folder.name.startswith("wl environment checks "))
        self.temp.cleanup()

    def test_historical_driver_assignment(self):
        for name in environments.MCALENS_DRIVERS:
            self.assertEqual(environments.role_for_driver(Path("code/pipeline") / name), "mcalens")
        for name in ("run_gaussian_starlet_test.py", "run_square_starlet_comparison.py",
                     "run_weighted_band_fusion_test.py", "run_admm_gaussian_starlet_audit.py"):
            self.assertEqual(environments.role_for_driver(Path("code/pipeline") / name), "gob5")
        self.assertEqual(environments.role_for_driver(Path("code/glimpse/run_glimpse_test_final.py")), "glimpse")

    def test_missing_modopt_blocks_readiness_and_does_not_require_torch(self):
        imported = []
        paths = types.SimpleNamespace(mcalens_root=self.folder)
        def fake_import(name):
            imported.append(name)
            if name == "modopt":
                raise ModuleNotFoundError("No module named 'modopt'")
            if name == "pycs.astro.wl.mass_mapping":
                return types.SimpleNamespace(__file__=str(self.folder / "mass_mapping.py"),
                                             massmap2d=object, shear_data=object)
            return types.SimpleNamespace(__version__="fixture", __file__=str(self.folder / "fixture.py"))
        before = list(sys.path)
        try:
            with patch.object(environment_probe.importlib, "import_module", side_effect=fake_import):
                result = environment_probe.probe("mcalens", paths)
        finally:
            sys.path[:] = before
        self.assertFalse(result["ok"])
        self.assertIn("modopt", result["imports"]["modopt"]["error"])
        self.assertNotIn("torch", imported)

    def test_mcalens_probe_reports_the_actual_source(self):
        paths = types.SimpleNamespace(mcalens_root=self.folder)
        def fake_import(name):
            return types.SimpleNamespace(__version__="fixture", __file__=str(self.folder / "mass_mapping.py"),
                                         massmap2d=object, shear_data=object)
        before = list(sys.path)
        try:
            with patch.object(environment_probe.importlib, "import_module", side_effect=fake_import):
                result = environment_probe.probe("mcalens", paths)
        finally:
            sys.path[:] = before
        self.assertTrue(result["ok"])
        self.assertEqual(result["python_executable"], sys.executable)

    def test_pythonpath_does_not_leak_between_environments(self):
        paths = types.SimpleNamespace(code_root=self.folder / "code", mcalens_root=self.folder / "cosmostat")
        with patch.dict(os.environ, {"PYTHONPATH": "wrong-environment", "PYTHONHOME": "wrong-home"}):
            core = environments.process_environment(paths, "gob5")
            mc = environments.process_environment(paths, "mcalens")
        self.assertNotIn("PYTHONHOME", mc)
        self.assertNotIn("wrong-environment", mc["PYTHONPATH"])
        self.assertNotIn(str(paths.mcalens_root), core["PYTHONPATH"])
        self.assertIn(str(paths.mcalens_root), mc["PYTHONPATH"])

    def test_worker_runs_in_separate_selected_process_without_torch(self):
        # Real subprocess and production worker, with a tiny synthetic scientific driver.
        # A dispatch marker ensures the child went through mcalens_python, not the parent Python command.
        fixture = self.folder / "fixture repo"
        (fixture / "tools").mkdir(parents=True)
        (fixture / "code/pipeline").mkdir(parents=True)
        for relative in ("tools/mcalens_smoke_worker.py", "code/pipeline/project_paths.py"):
            shutil.copyfile(environments.ROOT / relative, fixture / relative)
        work = self.folder / "new outputs"
        work.mkdir()
        cfg = work / "paths.toml"
        values = {"data_root": str(self.folder / "raw"), "work_root": str(work),
                  "glimpse_root": str(self.folder / "glimpse"), "mcalens_root": str(self.folder / "cosmostat")}
        cfg.write_text("".join(f"{k}={json.dumps(v)}\n" for k, v in values.items()))
        shutil.copyfile(cfg, fixture / "config_paths.toml")
        fake_driver = '''import os, sys
from project_paths import PATHS
assert "torch" not in sys.modules
assert os.environ.get("WL_TEST_ROLE") == "mcalens"
BASE = PATHS.work_root / "test/fair_npz"
OUT = PATHS.work_root / "test/mcalens"
RECON_DIR = OUT / "recon"
PK_FILE = PATHS.prior_file
MASK_FILE = PATHS.data_root / "WIDE12H_bin2_2arcmin_mask.npy"
RECON_DIR.mkdir(parents=True)
def run_one(case):
    assert case == (0, 10, 0)
    output = RECON_DIR / "C0_R10_P0.npz"
    output.write_bytes(b"synthetic reconstruction fixture")
    return {"finite": True, "pid": os.getpid(), "case": list(case), "recon_file": str(output)}
'''
        (fixture / "code/pipeline/run_mcalens_test.py").write_text(fake_driver)
        pycs = Path(values["mcalens_root"]) / "pycs/astro/wl"
        pycs.mkdir(parents=True)
        (pycs / "mass_mapping.py").write_text("# synthetic source for subprocess routing test\n")
        # This same-name historical script took precedence in the v5 worker.
        (Path(values["mcalens_root"]) / "run_mcalens_test.py").write_text(
            'raise RuntimeError("Historical driver shadowed the packaged driver")\n')
        historical = Path(values["glimpse_root"]) / "test/mcalens/recon/C0_R10_P0.npz"
        historical.parent.mkdir(parents=True)
        historical.write_bytes(b"preserve historical result")
        dispatch = self.folder / "mcalens dispatch.py"
        dispatch.write_text('import os, subprocess, sys\nos.environ["WL_TEST_ROLE"]="mcalens"\n'
                            'raise SystemExit(subprocess.run([sys.executable,*sys.argv[1:]],env=os.environ.copy()).returncode)\n')
        settings = {"mcalens_python": [sys.executable, str(dispatch)]}
        with patch.object(server_smoke, "ROOT", fixture):
            result = server_smoke.run_mcalens_process(settings, cfg, work)
        self.assertTrue(result["ok"])
        self.assertNotEqual(result["result"]["pid"], os.getpid())
        self.assertEqual(result["command"][:2], settings["mcalens_python"])
        self.assertTrue(result["historical_output_check"]["unchanged"])
        self.assertEqual(historical.read_bytes(), b"preserve historical result")
        self.assertEqual(Path(result["driver"]["path"]), fixture / "code/pipeline/run_mcalens_test.py")

    def test_launcher_dry_run_routes_three_environments(self):
        cfg = self.folder / "environments.toml"
        cfg.write_text(f'gob5_python={json.dumps([sys.executable])}\n'
                       'mcalens_python=["MC-PYTHON"]\nglimpse_prefix=["GL-ENV"]\n')
        for name, expected in (("run_mcalens_test.py", ["MC-PYTHON"]),
                               ("run_square_starlet_comparison.py", [sys.executable]),
                               ("run_glimpse_test_final.py", ["GL-ENV", sys.executable])):
            result = subprocess.run([sys.executable, "-B", str(environments.ROOT / "tools/run_pipeline.py"),
                                     "--environments", str(cfg), "--dry-run", name],
                                    text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout)["command"][:len(expected)], expected)

    def test_smoke_parent_has_no_mcalens_driver_import(self):
        tree = ast.parse((environments.ROOT / "tools/server_smoke.py").read_text(encoding="utf-8"))
        forbidden = {"run_mcalens_test", "pycs.astro.wl.mass_mapping", "modopt", "seaborn"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                if node.args and isinstance(node.args[0], ast.Constant):
                    self.assertNotIn(node.args[0].value, forbidden)


if __name__ == "__main__":
    unittest.main(verbosity=2)
