#!/usr/bin/env python3
"""
Verify a trimmed ShareGPT file against its original.

Proves three things with no assumption about where differences should be:
  1. The only differing fields are `value`s of `human` turns.
  2. Every other field (id, from, gpt/system values, extra fields) is identical.
  3. Every trimmed human value equals exactly the first N characters of the
     original, and every untrimmed human value is unchanged.

Usage:
    python3 verify_trim.py ORIGINAL.json TRIMMED.json -n 1024

Note: this loads BOTH files fully into memory (peak ~ a few GB for the 672 MB
ShareGPT_V3 file x2). Run on a box with enough RAM, or sample first.
"""
import argparse
import json
import sys


def deep_diff(a, b, path, diffs):
    """Record the path of every differing leaf between a and b."""
    if type(a) is not type(b):
        diffs.append((path, "type")); return
    if isinstance(a, dict):
        if a.keys() != b.keys():
            diffs.append((path, "keys")); return
        for k in a:
            deep_diff(a[k], b[k], path + [k], diffs)
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append((path, "len")); return
        for i, (x, y) in enumerate(zip(a, b)):
            deep_diff(x, y, path + [i], diffs)
    elif a != b:
        diffs.append((path, "value"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("original")
    p.add_argument("trimmed")
    p.add_argument("-n", "--num-chars", type=int, default=1024)
    p.add_argument("--role", default="human",
                   help="role whose values were trimmed (default: human)")
    args = p.parse_args()
    N = args.num_chars

    with open(args.original, encoding="utf-8") as f:
        orig = json.load(f)
    with open(args.trimmed, encoding="utf-8") as f:
        trim = json.load(f)

    if len(orig) != len(trim):
        sys.exit(f"FAIL: top-level length differs ({len(orig)} vs {len(trim)})")

    # 1. Generic deep-diff: locate every difference.
    diffs = []
    deep_diff(orig, trim, [], diffs)

    # 2. Every diff must be a `value` of a trimmed `role` turn, original > N chars.
    bad = []
    for path, kind in diffs:
        ok = (kind == "value" and len(path) == 4
              and path[1] == "conversations" and path[3] == "value")
        if ok:
            i, j = path[0], path[2]
            ov = orig[i]["conversations"][j].get("value")
            nv = trim[i]["conversations"][j].get("value")
            ok = (orig[i]["conversations"][j].get("from") == args.role
                  and isinstance(ov, str) and len(ov) > N
                  and nv == ov[:N] and len(nv) == N)
        if not ok:
            bad.append((path, kind))

    # 3. Universal invariants across every turn.
    role_total = role_trimmed = other_total = 0
    maxlen = 0
    for i, e in enumerate(orig):
        te = trim[i]
        if e.get("id") != te.get("id"):
            sys.exit(f"FAIL: id changed at index {i}")
        for j, t in enumerate(e["conversations"]):
            tt = te["conversations"][j]
            if t.get("from") != tt.get("from"):
                sys.exit(f"FAIL: 'from' changed at {i},{j}")
            ov, nv = t.get("value"), tt.get("value")
            if t.get("from") == args.role:
                role_total += 1
                if isinstance(ov, str):
                    maxlen = max(maxlen, len(ov))
                    if nv != ov[:N]:
                        sys.exit(f"FAIL: {args.role} value != first {N} chars at {i},{j}")
                    if len(ov) > N:
                        role_trimmed += 1
                elif nv != ov:
                    sys.exit(f"FAIL: non-string {args.role} value mutated at {i},{j}")
            else:
                other_total += 1
                if nv != ov:
                    sys.exit(f"FAIL: non-{args.role} value changed at {i},{j}")

    print(f"conversations            : {len(orig)}")
    print(f"differing leaves total   : {len(diffs)}")
    print(f"diffs outside a prompt   : {len(bad)}")
    print(f"{args.role} turns total      : {role_total}")
    print(f"{args.role} turns trimmed    : {role_trimmed}")
    print(f"other turns (identical)  : {other_total}")
    print(f"longest original prompt  : {maxlen} chars")

    if bad:
        sys.exit(f"FAIL: {len(bad)} diff(s) are NOT trimmed prompts, e.g. {bad[:3]}")
    if len(diffs) != role_trimmed:
        sys.exit(f"FAIL: diff count {len(diffs)} != trimmed count {role_trimmed}")

    print("\nPASS: files match exactly except trimmed prompts; "
          f"every trimmed prompt == first {N} chars of the original.")


if __name__ == "__main__":
    main()