"""shared helpers of the scripts: argument parsing, output folder, csv/markdown writing"""
import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jog_eval import config as C  # noqa: E402


def parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument('--basins', nargs='+', default=list(C.BASINS))
    p.add_argument('--out', default=str(C.OUT_ROOT))
    p.add_argument('--definition', default='both', choices=['paper', 'published_code', 'both'])
    return p


def outdir(args, name):
    d = Path(args.out) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_csv(fn, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(fn, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(fn, obj):
    Path(fn).write_text(json.dumps(obj, indent=2, default=str))


def definitions(args):
    return ['paper', 'published_code'] if args.definition == 'both' else [args.definition]
