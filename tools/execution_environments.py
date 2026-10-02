"""Process dispatch only; no package installation or scientific configuration."""
import os
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MCALENS_DRIVERS = {
    "generate_mcalens_validation_maps.py", "run_mcalens_validation.py",
    "run_mcalens_test.py", "build_fair_validation_pk.py",
}


def load_environments(file=None):
    path = Path(file or ROOT / "config_environments.server.toml").expanduser().absolute()
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    if set(settings) != {"gob5_python", "mcalens_python", "glimpse_prefix"}:
        raise ValueError("Expected gob5_python, mcalens_python and glimpse_prefix commands")
    for key, command in settings.items():
        if not isinstance(command, list) or (not command and key != "glimpse_prefix"):
            raise ValueError(f"{key} must be an argument array")
        if any(not isinstance(v, str) or not v.strip() for v in command):
            raise ValueError(f"{key} has an empty/non-string argument")
        settings[key] = [str(Path(v).expanduser()) if v.startswith("~") else v for v in command]
    # The core process is a directly invoked interpreter; no shell activation needed.
    if len(settings["gob5_python"]) != 1:
        raise ValueError("gob5_python must contain the direct interpreter path")
    if os.environ.get("WL_PYTHON"):
        settings["gob5_python"] = [str(Path(os.environ["WL_PYTHON"]).expanduser())]
    return settings


def role_for_driver(path):
    if path.name in MCALENS_DRIVERS:
        return "mcalens"
    if path.parent.name == "glimpse":
        return "glimpse"
    return "gob5"


def python_command(settings, role):
    if role == "mcalens":
        return list(settings["mcalens_python"])
    if role == "glimpse":
        return [*settings["glimpse_prefix"], *settings["gob5_python"]]
    if role == "gob5":
        return list(settings["gob5_python"])
    raise ValueError(f"Unknown role: {role}")


def process_environment(paths, role):
    env = os.environ.copy()
    # Never import packages from the caller's unrelated Python environment.
    env.pop("PYTHONHOME", None)
    env.pop("VIRTUAL_ENV", None)
    roots = [str(paths.code_root)]
    if role == "mcalens":
        roots.insert(0, str(paths.mcalens_root))
    env["PYTHONPATH"] = os.pathsep.join(roots)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env
