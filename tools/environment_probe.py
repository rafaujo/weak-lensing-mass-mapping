"""Actually import dependencies in the selected interpreter, without reconstruction."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/pipeline"))
from project_paths import load_paths


def probe(kind, paths):
    modules = {
        "gob5": ["numpy", "scipy", "pandas", "torch", "pywt", "astropy", "matplotlib"],
        "mcalens": ["numpy", "scipy", "pandas", "astropy", "matplotlib", "seaborn", "modopt",
                    "pycs.astro.wl.mass_mapping"],
    }[kind]
    if kind == "mcalens":
        sys.path.insert(0, str(paths.mcalens_root))
    result = {"kind": kind, "python_executable": sys.executable, "python_version": sys.version,
              "prefix": sys.prefix, "base_prefix": sys.base_prefix, "imports": {}, "ok": True}
    for name in modules:
        try:
            module = importlib.import_module(name)
            item = {"ok": True, "version": str(getattr(module, "__version__", "unknown")),
                    "file": getattr(module, "__file__", None)}
            if name == "pycs.astro.wl.mass_mapping":
                item["classes_present"] = all(hasattr(module, n) for n in ("massmap2d", "shear_data"))
                item["configured_checkout"] = Path(item["file"]).resolve().is_relative_to(paths.mcalens_root)
                item["ok"] = item["classes_present"] and item["configured_checkout"]
            result["ok"] = result["ok"] and item["ok"]
            result["imports"][name] = item
        except Exception as exc:
            result["ok"] = False
            result["imports"][name] = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("gob5", "mcalens"), required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.kind, load_paths(args.paths))
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
