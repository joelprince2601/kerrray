"""Compare regenerated paper data with the committed originals.

For every JSON file in paper/data/ this loads the version at a git revision
(default 7788ef7, the commit the re-run started from) and the file on disk,
walks both, and reports numeric leaves that differ, ignoring run times,
timestamps and provenance fields. Exit status is 0 in all cases; the report
is the output.
Run: ./.venv/Scripts/python.exe scripts/audit/compare_reproduction.py [rev]
"""
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REV = sys.argv[1] if len(sys.argv) > 1 else "7788ef7"
IGNORE = {"runtime_s", "git_commit", "git_dirty", "hardware", "timestamp", "runtime_repeats_s", "seconds", "elapsed_s"}


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in IGNORE or "runtime" in k or k.endswith("_s"):
                continue
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def main() -> None:
    for f in sorted((ROOT / "paper" / "data").glob("*.json")):
        rel = f.relative_to(ROOT).as_posix()
        try:
            old = json.loads(subprocess.run(["git", "show", f"{REV}:{rel}"], cwd=ROOT, capture_output=True, check=True, text=True, encoding="utf-8").stdout)
        except subprocess.CalledProcessError:
            print(f"{rel}: not in {REV}")
            continue
        new = json.loads(f.read_text(encoding="utf-8"))
        a, b = dict(leaves(old)), dict(leaves(new))
        only_old, only_new = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        diffs, worst = [], 0.0
        for k in sorted(set(a) & set(b)):
            x, y = a[k], b[k]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                if (isinstance(x, float) and math.isnan(x)) and (isinstance(y, float) and math.isnan(y)):
                    continue
                if x != y:
                    rel_d = abs(x - y) / max(abs(x), abs(y), 1e-300)
                    worst = max(worst, rel_d)
                    diffs.append((k, x, y, rel_d))
            elif x != y:
                diffs.append((k, x, y, float("nan")))
        status = "IDENTICAL" if not diffs and not only_old and not only_new else f"{len(diffs)} differing values (max relative difference {worst:.2e})"
        print(f"{rel}: {len(a)} values compared; {status}")
        for k, x, y, r in diffs[:25]:
            print(f"    {k}: {x!r} -> {y!r} (rel {r:.2e})")
        if len(diffs) > 25:
            print(f"    ... {len(diffs) - 25} more")
        if only_old or only_new:
            print(f"    keys only in {REV}: {len(only_old)}; only in the re-run: {len(only_new)}")


if __name__ == "__main__":
    main()
