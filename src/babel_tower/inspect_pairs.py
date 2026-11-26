#!/usr/bin/env python3
"""Look at alignment output without reading everything.

Sample pairs in a similarity range (e.g. to judge where the quality cut-off should be):
    python inspect_pairs.py sample data/aligned/pooh.pairs.jsonl --lo 0.5 --hi 0.6 -n 12
    python inspect_pairs.py sample data/aligned/pooh.pairs.jsonl --type 2-2 -n 12

Compare two alignment runs (e.g. different --group-penalty or --skip-cost) and show only
the pairs that differ:
    python inspect_pairs.py compare data/aligned/pooh.pairs.jsonl data/aligned/pooh_gp02.pairs.jsonl -n 8
"""
import argparse
import json
import random
from collections import Counter


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def key(p):
    return (tuple(p["en_ids"]), tuple(p["hu_ids"]))


def clip(s, w=100):
    return s if len(s) <= w else s[: w - 1] + "…"


def show(p):
    print(f"  [{p['type']} sim={p['sim']:.2f}]")
    print(f"     EN {clip(p['en'])}")
    print(f"     HU {clip(p['hu'])}")


def cmd_sample(a):
    ps = [p for p in load(a.pairs)
          if a.lo <= p["sim"] < a.hi and (a.type is None or p["type"] == a.type)]
    random.seed(a.seed)
    print(f"{len(ps)} pairs match")
    for p in random.sample(ps, min(a.n, len(ps))):
        show(p)


def cmd_compare(a):
    A, B = load(a.a), load(a.b)
    keys_a, keys_b = {key(p) for p in A}, {key(p) for p in B}
    only_a = [p for p in A if key(p) not in keys_b]
    only_b = [p for p in B if key(p) not in keys_a]
    print(f"A: {len(A)} pairs   B: {len(B)} pairs   identical in both: {len(A) - len(only_a)}")
    random.seed(a.seed)
    for name, ps in (("only in A", only_a), ("only in B", only_b)):
        types = Counter(p["type"] for p in ps)
        mean = sum(p["sim"] for p in ps) / len(ps) if ps else 0.0
        print(f"--- {name}: {len(ps)} pairs, mean sim {mean:.2f}, types {dict(sorted(types.items()))} ---")
        for p in random.sample(ps, min(a.n, len(ps))):
            show(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("pairs")
    s.add_argument("--lo", type=float, default=0.0)
    s.add_argument("--hi", type=float, default=1.01)
    s.add_argument("--type", default=None, help="e.g. 2-2")
    s.add_argument("-n", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_sample)

    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("-n", type=int, default=8)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(fn=cmd_compare)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()