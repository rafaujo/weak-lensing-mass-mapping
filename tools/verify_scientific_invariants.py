"""Verify that only the path edits planned against v2 changed scientific sources."""
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sha = lambda data: hashlib.sha256(data).hexdigest()
dump = lambda node: ast.dump(node, include_attributes=False)


def at(tree, address):
    for key in address:
        tree = tree[key] if isinstance(key, int) else getattr(tree, key)
    return tree


def replace(tree, address, node):
    parent = at(tree, address[:-1])
    if isinstance(address[-1], int):
        parent[address[-1]] = node
    else:
        setattr(parent, address[-1], node)


def verify(root=ROOT):
    folder = root / "provenance" / "portability"
    plan = json.loads((folder / "planned_changes.json").read_text(encoding="utf-8"))
    records = {r["file"]: r for r in json.loads((folder / "transformed_files.json").read_text(encoding="utf-8"))}
    checked = []
    for entry in plan["files"]:
        path = root / entry["file"]
        tree = ast.parse(path.read_bytes())
        compile(tree, str(path), "exec")
        numeric = [repr(n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant)
                   and isinstance(n.value, (int, float, complex, bool))]
        numeric_ok = sha(json.dumps(numeric).encode()) == entry["numeric_literal_sha256"]
        restored = copy.deepcopy(tree)
        imports = [n for n in restored.body if isinstance(n, ast.ImportFrom) and n.module == "project_paths"]
        import_ok = (len(imports) == 1 and dump(imports[0]) == dump(ast.parse(entry["added_import"]).body[0])) if entry["changes"] else not imports
        if not import_ok:
            raise ValueError(f"Unexpected path import: {entry['file']}")
        if imports:
            restored.body.remove(imports[0])
        for change in entry["changes"]:
            expected = ast.parse(change["after"], mode="eval").body
            if dump(at(restored, change["address"])) != dump(expected):
                raise ValueError(f"Unplanned change at {entry['file']}:{change['line']}")
            replace(restored, change["address"], ast.parse("(\n" + change["before"] + "\n)", mode="eval").body)
        ast_ok = sha(dump(restored).encode()) == entry["baseline_ast_sha256"]
        bytes_ok = sha(path.read_bytes()) == records[entry["file"]]["portable_sha256"]
        checked.append({"file": entry["file"], "path_edits": len(entry["changes"]),
                        "numeric_literals": len(numeric), "numeric_literals_identical": numeric_ok,
                        "entire_ast_identical_after_reversing_planned_paths": ast_ok,
                        "bytes_match_recorded_refactoring": bytes_ok,
                        "pass": numeric_ok and ast_ok and bytes_ok})
    transformed = {e["file"] for e in plan["files"] if e["changes"]}
    unchanged = []
    with (root / "provenance" / "curation_manifest.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["decision"] != "keep" or row["destination"] in transformed:
                continue
            unchanged.append({"file": row["destination"],
                              "pass": sha((root / row["destination"]).read_bytes()) == row["source_sha256"]})
    return {"baseline_archive_sha256": plan["baseline_archive_sha256"],
            "scientific_scripts": len(checked), "path_edits": sum(c["path_edits"] for c in checked),
            "numeric_literals_checked": sum(c["numeric_literals"] for c in checked),
            "unchanged_original_files": len(unchanged), "scripts": checked,
            "unchanged_originals": unchanged,
            "failed": sum(not c["pass"] for c in checked + unchanged),
            "scope": "Source-structure and input-integrity preservation, not numerical solver equivalence across environments."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"scripts", "unchanged_originals"}}, indent=2))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
