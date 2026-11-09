#!/usr/bin/env python3
"""Local scorer mirroring the bounty's rules (log score floored at 1e-4).

Usage: python3 tools/score_local.py <forecast.json> <answers.json>

answers.json: {"<date>": {"<symbol>": <return>, ...}, ...} for the scored days.
"""

import json
import math
import sys

FLOOR = 1e-4


def bin_of(r, edges):
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if edges[mid] <= r:
            lo = mid + 1
        else:
            hi = mid
    return lo


def score_day(day, actual, places, edges):
    chart = sorted(actual, key=lambda s: -actual[s])
    named = [dict() for _ in range(places)]
    sums = [0.0] * places
    for row in day["probabilities"]:
        named[row["place"] - 1][row["symbol"]] = row["p"]
        sums[row["place"] - 1] += row["p"]
    order_score = 0.0
    n = min(places, len(chart))
    for k in range(n):
        sym = chart[k]
        at = named[k]
        if sym in at:
            p = at[sym]
        else:
            unnamed = sum(1 for c in chart if c not in at)
            p = max(0.0, 1.0 - sums[k]) / unnamed if unnamed else 0.0
        order_score += math.log(max(p, FLOOR))
    bins = len(edges) + 1
    sized = {}
    ssum = {}
    for row in day["sizes"]:
        sized.setdefault(row["symbol"], {})[row["bin"]] = row["p"]
        ssum[row["symbol"]] = ssum.get(row["symbol"], 0.0) + row["p"]
    size_score = 0.0
    for sym, r in actual.items():
        b = bin_of(r, edges)
        at = sized.get(sym, {})
        if b in at:
            p = at[b]
        else:
            unnamed = bins - len(at)
            p = max(0.0, 1.0 - ssum.get(sym, 0.0)) / unnamed if unnamed else 0.0
        size_score += math.log(max(p, FLOOR))
    top = day["order"]
    head = chart[: len(top)]
    agreement = 0
    for i, sym in enumerate(top):
        if sym in head:
            agreement += len(top) - abs(i - head.index(sym))
    return order_score, size_score, agreement


def main():
    forecast = json.load(open(sys.argv[1], encoding="utf-8"))
    answers = json.load(open(sys.argv[2], encoding="utf-8"))
    edges = [round(-0.1 + 0.005 * i, 3) for i in range(41)]
    total_o = total_s = 0.0
    agreements = []
    for day in forecast["days"]:
        actual = answers.get(day["date"])
        if not actual:
            continue
        places = len(day["order"])
        o, s, a = score_day(day, actual, places, edges)
        total_o += o
        total_s += s
        agreements.append(a)
        print(f"{day['date']}: order {o:.6f} size {s:.6f} agreement {a}")
    print(f"TOTAL log score {total_o + total_s:.6f} (order {total_o:.6f}, size {total_s:.6f}), "
          f"mean agreement {sum(agreements) / max(1, len(agreements)):.1f}")


if __name__ == "__main__":
    main()
