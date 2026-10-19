#!/usr/bin/env python3
"""Compare two probe4 teacher-forced dumps: top-1 agreement % and max |dlogprob|.

Usage: compare_tf.py <a.jsonl> <b.jsonl>
Both are probe4 output (kind="tf", pos, top=[[token_id, logprob], ...]).
"""
import json
import sys


def load(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        j = json.loads(line)
        if j.get("kind") != "tf" or not j.get("top"):
            continue
        out[int(j["pos"])] = {t: float(lp) for t, lp in j["top"]}
    return out


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    pos = sorted(set(a) & set(b))
    if not pos:
        print(json.dumps({"error": "no common positions"}))
        return
    agree = 0
    dmax = 0.0
    dsum = 0.0
    n = 0
    missing = 0
    for p in pos:
        ta = max(a[p], key=a[p].get)
        tb = max(b[p], key=b[p].get)
        if ta == tb:
            agree += 1
        if ta in b[p]:
            d = abs(a[p][ta] - b[p][ta])
            dmax = max(dmax, d)
            dsum += d
            n += 1
        else:
            missing += 1
    print(json.dumps({
        "positions": len(pos),
        "top1_agreement_pct": round(100.0 * agree / len(pos), 2),
        "max_abs_dlogprob": round(dmax, 6),
        "mean_abs_dlogprob": round(dsum / n, 8) if n else None,
        "positions_missing_top1_in_b": missing,
    }))


if __name__ == "__main__":
    main()