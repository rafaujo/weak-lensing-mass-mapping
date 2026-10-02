"""Integration tests for central paths/launcher and negative tests of source guards."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
from project_paths import load_paths
from verify_scientific_invariants import verify


class PortabilityChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wl path checks ")
        self.folder = Path(self.temp.name)
        self.config = self.folder / "config with spaces.toml"
        self.config.write_text('data_root="raw inputs"\nwork_root="generated outputs"\n'
                               'glimpse_root="glimpse checkout"\nmcalens_root="cosmostat checkout"\n', encoding="utf-8")

    def tearDown(self):
        self.assertTrue(self.folder.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.assertTrue(self.folder.name.startswith("wl path checks "))
        self.temp.cleanup()

    def test_relative_paths_and_spaces(self):
        paths = load_paths(self.config)
        self.assertEqual(paths.data_root, self.folder / "raw inputs")
        self.assertEqual(paths.work_root, self.folder / "generated outputs")
        self.assertEqual(paths.prior_file, ROOT / "config/priors/fair_validation_pk_mean.npy")
        self.assertFalse(paths.work_root.exists())  # reading config performs no writes

    def test_default_config_is_independent_of_cwd(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/run_pipeline.py"),
                                 "--dry-run", "run_mcalens_test.py"], cwd=self.folder,
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["paths"]["data_root"], str(ROOT / "data"))

    def test_custom_config_and_existing_arguments(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/run_pipeline.py"),
                                 "--paths", str(self.config), "--dry-run",
                                 "run_square_starlet_comparison.py", "--split", "validation"],
                                cwd=self.folder, capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report["arguments"], ["--split", "validation"])
        self.assertEqual(report["paths"]["work_root"], str(self.folder / "generated outputs"))

    def test_unknown_configuration_key_rejected(self):
        with self.config.open("a", encoding="utf-8") as stream:
            stream.write('lambda = 0.06\n')
        with self.assertRaises(ValueError):
            load_paths(self.config)

    def test_overlapping_output_root_rejected(self):
        self.config.write_text(self.config.read_text().replace('work_root="generated outputs"',
                                                              'work_root="raw inputs/generated"'))
        with self.assertRaises(ValueError):
            load_paths(self.config)

    def test_launch_and_spawn_inherit_paths(self):
        # A tiny synthetic driver validates real subprocess and spawn behavior;
        # none of the scientific benchmark drivers is imported by this test.
        repo = self.folder / "fake repo"
        (repo / "tools").mkdir(parents=True)
        (repo / "code/pipeline").mkdir(parents=True)
        (repo / "code/glimpse").mkdir(parents=True)
        shutil.copyfile(ROOT / "tools/run_pipeline.py", repo / "tools/run_pipeline.py")
        shutil.copyfile(ROOT / "code/pipeline/project_paths.py", repo / "code/pipeline/project_paths.py")
        shutil.copyfile(self.config, repo / "config_paths.toml")
        (repo / "cosmostat checkout").mkdir()
        (repo / "cosmostat checkout/local_probe.py").write_text('VALUE = "from configured checkout"\n')
        (repo / "code/glimpse/probe.py").write_text('''import json, sys, multiprocessing
from concurrent.futures import ProcessPoolExecutor
from project_paths import PATHS
def worker():
    from local_probe import VALUE
    return [VALUE, str(PATHS.work_root)]
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        print(json.dumps({"arguments":sys.argv[1:],"child":pool.submit(worker).result()}))
''', encoding="utf-8")
        result = subprocess.run([sys.executable, "-B", str(repo / "tools/run_pipeline.py"),
                                 "probe.py", "--split", "test"], cwd=self.folder,
                                capture_output=True, text=True, check=True, timeout=30)
        report = json.loads(result.stdout)
        self.assertEqual(report["arguments"], ["--split", "test"])
        self.assertEqual(report["child"], ["from configured checkout", str(repo / "generated outputs")])

    def test_source_guards_detect_changes(self):
        self.assertEqual(verify()["failed"], 0)
        repo = self.folder / "source guard fixture"
        shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns("__pycache__", ".git", "work", "data"))
        path = repo / "code/pipeline/run_weighted_band_fusion_test.py"
        original = path.read_text(encoding="utf-8")
        for old, new in [("A1 = 0.75", "A1 = 0.50"), ("J = 4", "J = 5")]:
            self.assertIn(old, original)
            path.write_text(original.replace(old, new, 1), encoding="utf-8")
            report = verify(repo)
            changed = next(r for r in report["scripts"] if r["file"] == path.relative_to(repo).as_posix())
            self.assertFalse(changed["entire_ast_identical_after_reversing_planned_paths"])
            self.assertFalse(changed["numeric_literals_identical"])
            path.write_text(original, encoding="utf-8")
        path.write_text(original.replace("PATHS.work_root", "PATHS.data_root", 1), encoding="utf-8")
        with self.assertRaises(ValueError):
            verify(repo)

    def test_historical_glimpse_layout_can_be_configured(self):
        self.config.write_text(self.config.read_text().replace('glimpse_root="glimpse checkout"',
                                                              'glimpse_root="generated outputs"'))
        paths = load_paths(self.config)
        self.assertEqual(paths.work_root, paths.glimpse_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PortabilityChecks)
    result = unittest.TestResult()
    suite.run(result)
    report = {"tests": result.testsRun,
              "failures": [{"test": str(t), "detail": e} for t, e in result.failures + result.errors],
              "scope": "Filesystem configuration, CLI, subprocess/spawn imports and source-guard rejection; no numerical solver run."}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
