"""Launch an original scientific driver with central paths and its existing CLI."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, help="Alternative config_paths.toml")
    parser.add_argument("--environments", type=Path, help="Route drivers to the recorded server interpreters")
    parser.add_argument("--list", action="store_true", help="List drivers without importing them")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths and command without executing")
    parser.add_argument("script", nargs="?", help="Driver filename, e.g. run_square_starlet_comparison.py")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Existing driver arguments")
    args = parser.parse_args()
    scripts = {p.name: p for p in (ROOT / "code").rglob("*.py")
               if p.name not in {"project_paths.py", "weak_lensing_pipeline.py"}}
    if args.list:
        print("\n".join(sorted(scripts)))
        return
    if args.script not in scripts:
        parser.error("Choose a driver from --list")
    if args.paths:
        os.environ["WL_PATHS_CONFIG"] = str(args.paths.expanduser().resolve())
    sys.path.insert(0, str(ROOT / "code" / "pipeline"))
    from project_paths import PATHS
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    command = [sys.executable, str(scripts[args.script]), *arguments]
    role = "current interpreter"
    if args.environments:
        from execution_environments import load_environments, role_for_driver, python_command, process_environment
        role = role_for_driver(scripts[args.script])
        command = [*python_command(load_environments(args.environments), role),
                   str(scripts[args.script]), *arguments]
    if args.dry_run:
        print(json.dumps({"driver": str(scripts[args.script]), "arguments": arguments,
                          "paths": PATHS.as_dict(), "environment": role, "command": command}, indent=2))
        return
    env = os.environ.copy()
    entries = [str(PATHS.code_root), str(PATHS.mcalens_root)]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    if args.environments:
        env = process_environment(PATHS, role)
    # A real script process preserves multiprocessing spawn semantics; workers
    # inherit the configured module roots and configuration-file environment.
    result = subprocess.run(command, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
