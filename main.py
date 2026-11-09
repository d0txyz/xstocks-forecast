#!/usr/bin/env python3
"""xStocks movers forecast (make.fun bounty xstocks-movers-forecast-01).

Reads /input/task.json and the hourly bars of each name, writes /output/forecast.json:
for every window day, a probability for each name at each place of the day's order by
return, a point guess at the order, and a probability for each name's return in the
task's half-point bins.

Model (standard library only, deterministic):

  * Each name's daily return is Student-t (nu = 4) with sd its forecast daily vol.
    The vol blends a per-name prior for the current regime with the realized vol of the
    hourly bars in the history (log-space shrinkage, the history's weight decaying with
    the horizon), then applies a per-day multiplier (scheduled macro releases, the
    weekend gap before a Monday).
  * The first window day's return runs from the previous session's close, and the
    history already holds the hours after that close. Their move is a known part of
    the day's return: it shifts the mean, and the remaining variance is reduced.
  * Returns co-move through one market factor: a multivariate Student-t with a common
    scale, so a wild day is wild for every name. Place probabilities come from a seeded
    Monte Carlo of that joint model; bin probabilities are the exact marginal t
    probabilities, so sizes and order come from the same distribution.

Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and
/output:  MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

from __future__ import annotations

import json
import math
import os
import random
from datetime import date, datetime, timedelta, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

HOUR_MS = 3_600_000
MICRO = 1_000_000
SEED = 20_261_110
DRAWS = 100_000

# Student-t degrees of freedom for daily returns (fat tails plus vol uncertainty).
NU = 4.0

# Prior daily close-to-close vol per name for the week of 2026-11-10: a high-vol tape
# (VIX near 28 after the 5% Nasdaq drop of 2026-11-05 and the 2.7% rebound of 11-09).
# Judgment from each name's usual vol, scaled for the regime, plus name-specific news
# flow (NVDA: board probe headlines and earnings on 11-17; SPCX: a young listing moving
# 6% to 9% a day; MSTR: bitcoin below its average cost).
PRIOR_VOL = {
    "AAPLX": 0.020,
    "MSFTX": 0.019,
    "GOOGLX": 0.024,
    "AMZNX": 0.025,
    "METAX": 0.027,
    "NVDAX": 0.050,
    "TSLAX": 0.045,
    "MSTRX": 0.055,
    "SPCXX": 0.050,
}
DEFAULT_PRIOR_VOL = 0.030
# Log scores punish a vol set too low more than one set too high: a small inflation of
# every forecast vol is the robust choice under vol uncertainty.
VOL_INFLATE = 1.05
# A vol spike fades: each day further out, the vol eases toward a calmer tape.
REGIME_DECAY = 0.985

# Share of each name's variance that comes from the common market factor.
FACTOR_SHARE = {
    "AAPLX": 0.55,
    "MSFTX": 0.60,
    "GOOGLX": 0.55,
    "AMZNX": 0.60,
    "METAX": 0.50,
    "NVDAX": 0.50,
    "TSLAX": 0.40,
    "MSTRX": 0.30,
    "SPCXX": 0.30,
}
DEFAULT_FACTOR_SHARE = 0.40

# Weight of the history's realized vol against the prior (log space), and its decay per
# day of horizon: one session of hourly bars says most about the next day.
RV_WEIGHT = 0.40
RV_DECAY = 0.85
# Hourly returns beyond this many RMS are winsorized (pool glitches, thin hours).
WINSOR = 3.5
# Fraction of the post-close move that is expected to persist in the day's return.
POSTCLOSE_CARRY = 0.85
# Known scheduled releases that widen the day: date -> vol multiplier.
EVENT_MULT = {
    "2026-11-10": 1.08,  # October CPI, 08:30 New York
    "2026-11-13": 1.03,  # October PPI
}
MONDAY_MULT = 1.05  # a Monday's return spans the weekend
MIN_VOL = 0.006
MAX_VOL = 0.12

# Smoothing so no (name, place) is ever near the scoring floor.
PLACE_MIX = 0.01
BIN_FLOOR = 0.0002


# ----------------------------------------------------------------------------- input

def load_task() -> dict:
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_bars(file: str) -> list[dict]:
    path = os.path.join(INPUT_DIR, file)
    if not os.path.isfile(path):
        return []
    bars: dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                t = int(row["t"])
                c = float(int(row["c"]))
                o = float(int(row["o"])) if row.get("o") is not None else c
            except (KeyError, ValueError, TypeError):
                continue
            if c <= 0 or o <= 0:
                continue
            bars[t] = {"t": t, "o": o, "c": c}
    return [bars[t] for t in sorted(bars)]


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


# ------------------------------------------------------------------------ statistics

def hourly_log_returns(bars: list[dict]) -> list[float]:
    out: list[float] = []
    for a, b in zip(bars, bars[1:]):
        if b["t"] - a["t"] == HOUR_MS:
            out.append(math.log(b["c"] / a["c"]))
    return out


def winsorize(rets: list[float]) -> list[float]:
    if len(rets) < 4:
        return rets
    rms = math.sqrt(sum(r * r for r in rets) / len(rets))
    cap = WINSOR * rms
    if cap <= 0:
        return rets
    return [max(-cap, min(cap, r)) for r in rets]


def realized_daily_vol(rets: list[float]) -> float | None:
    """Daily vol from one session's hourly returns: the sum of squares, scaled gently for
    the hours of the 24-hour session the history does not cover (quiet hours)."""
    rets = winsorize(rets)
    n = len(rets)
    if n < 6:
        return None
    var = sum(r * r for r in rets)
    missing = max(0, 24 - n)
    var *= 1.0 + 0.5 * missing / n
    return math.sqrt(var)


def blend_vol(prior: float, rv: float | None, weight: float) -> float:
    if rv is None or rv <= 0:
        return prior
    lv = (1.0 - weight) * math.log(prior) + weight * math.log(rv)
    return math.exp(lv)


# ----------------------------------------------------------------- Student-t maths

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    max_it = 300
    eps = 3e-15
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, nu: float) -> float:
    """CDF of Student's t with nu degrees of freedom."""
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    x = nu / (nu + t * t)
    tail = 0.5 * betainc(nu / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_scale(sd: float, nu: float) -> float:
    """Scale of a Student-t with the given standard deviation."""
    return sd * math.sqrt((nu - 2.0) / nu)


def bin_probabilities(mean: float, sd: float, edges: list[float], nu: float) -> list[float]:
    s = t_scale(sd, nu)
    cdf = [t_cdf((e - mean) / s, nu) for e in edges]
    probs = [cdf[0]]
    for i in range(1, len(edges)):
        probs.append(max(0.0, cdf[i] - cdf[i - 1]))
    probs.append(max(0.0, 1.0 - cdf[-1]))
    return probs


# ----------------------------------------------------------------------- the model

def day_index_multiplier(day: dict) -> float:
    mult = EVENT_MULT.get(day["date"], 1.0)
    try:
        d = date.fromisoformat(day["date"])
        if d.weekday() == 0:
            mult *= MONDAY_MULT
    except ValueError:
        pass
    return mult


def previous_close_instant(day: dict, task: dict) -> datetime | None:
    """The UTC instant the first window day's return runs from: the previous session's
    close TWAP end, from the task's window, else from the history's end."""
    try:
        twap_until = parse_iso(day["twap_until"])
        twap_from = parse_iso(day["twap_from"])
        session_len = twap_until - twap_from
        prev = date.fromisoformat(day["previous"])
        d0 = date.fromisoformat(day["date"])
        # The close of the previous session day is at the same clock time as this day's.
        prev_until = twap_until - timedelta(days=(d0 - prev).days)
        return prev_until if session_len.total_seconds() > 0 else None
    except (KeyError, ValueError, TypeError):
        return None


def post_close_move(bars: list[dict], close_at: datetime | None) -> float:
    """Log move from the previous session's close (estimated TWAP) to the history's last
    bar, or 0 when the history ends at or before that close."""
    if close_at is None or not bars:
        return 0.0
    close_ms = int(close_at.timestamp() * 1000)
    by_t = {b["t"]: b for b in bars}
    last = bars[-1]
    if last["t"] <= close_ms:
        return 0.0
    close_bar = by_t.get(close_ms)
    if close_bar is None:
        return 0.0
    # The 30-minute TWAP before the close sits between the hour's first and last five
    # minutes: a Brownian-bridge estimate weights the close 3 to 1.
    twap = 0.25 * close_bar["o"] + 0.75 * close_bar["c"]
    hours_after = (last["t"] - close_ms) / HOUR_MS
    if hours_after > 6:
        return 0.0
    return math.log(last["c"] / twap)


def forecast(task: dict) -> dict:
    universe = task["universe"]
    symbols = [u["symbol"] for u in universe]
    n = len(symbols)
    places = int(task.get("places", min(10, n)))
    edges = [float(e) for e in task["bin_edges"]]
    days = task["window"]["days"]

    bars = {u["symbol"]: load_bars(u["file"]) for u in universe}
    prior = [PRIOR_VOL.get(s, DEFAULT_PRIOR_VOL) for s in symbols]
    rho = [FACTOR_SHARE.get(s, DEFAULT_FACTOR_SHARE) for s in symbols]
    rv = [realized_daily_vol(hourly_log_returns(bars[s])) for s in symbols]

    first_close = previous_close_instant(days[0], task) if days else None
    moves = [post_close_move(bars[s], first_close) for s in symbols]
    hours_known = 0.0
    if first_close is not None:
        last_ts = [bars[s][-1]["t"] for s in symbols if bars[s]]
        if last_ts:
            hours_known = max(0.0, (max(last_ts) - int(first_close.timestamp() * 1000)) / HOUR_MS)
            if hours_known > 6:
                hours_known = 0.0

    rng = random.Random(SEED)
    out_days = []
    for d_idx, day in enumerate(days):
        weight = RV_WEIGHT * (RV_DECAY ** d_idx)
        mult = day_index_multiplier(day) * VOL_INFLATE * (REGIME_DECAY ** d_idx)
        sd = []
        mean = []
        for i in range(n):
            v = blend_vol(prior[i], rv[i], weight) * mult
            if d_idx == 0 and hours_known > 0:
                # Post-close hours are quiet: count each as a third of an average hour.
                known_share = min(0.5, (hours_known / 3.0) / 24.0)
                v *= math.sqrt(max(0.3, 1.0 - known_share))
                mean.append(POSTCLOSE_CARRY * moves[i])
            else:
                mean.append(0.0)
            sd.append(min(MAX_VOL, max(MIN_VOL, v)))

        # Sizes: exact marginal t probabilities, floored and renormalized.
        sizes = []
        for i, s in enumerate(symbols):
            probs = bin_probabilities(mean[i], sd[i], edges, NU)
            probs = [max(BIN_FLOOR, p) for p in probs]
            tot = sum(probs)
            probs = [p / tot for p in probs]
            for b, p in enumerate(probs):
                u = math.floor(p * MICRO)
                if u > 0:
                    sizes.append({"symbol": s, "bin": b, "p": u / MICRO})

        # Order: Monte Carlo of the joint multivariate t.
        scale = [t_scale(sd[i], NU) for i in range(n)]
        load_f = [math.sqrt(rho[i]) for i in range(n)]
        load_e = [math.sqrt(1.0 - rho[i]) for i in range(n)]
        counts = [[0] * n for _ in range(places)]
        rank_sum = [0.0] * n
        half_nu = NU / 2.0
        for _ in range(DRAWS):
            w = math.sqrt(NU / rng.gammavariate(half_nu, 2.0))
            zf = rng.gauss(0.0, 1.0)
            draw = [mean[i] + scale[i] * w * (load_f[i] * zf + load_e[i] * rng.gauss(0.0, 1.0)) for i in range(n)]
            order = sorted(range(n), key=lambda i: (-draw[i], i))
            for k in range(places):
                counts[k][order[k]] += 1
            for k, i in enumerate(order):
                rank_sum[i] += k

        probabilities = []
        for k in range(places):
            row = [(1.0 - PLACE_MIX) * counts[k][i] / DRAWS + PLACE_MIX / n for i in range(n)]
            for i, s in enumerate(symbols):
                u = math.floor(row[i] * MICRO)
                probabilities.append({"symbol": s, "place": k + 1, "p": u / MICRO})

        by_expected_rank = sorted(range(n), key=lambda i: (rank_sum[i], symbols[i]))
        order_guess = [symbols[i] for i in by_expected_rank[:places]]

        out_days.append({
            "date": day["date"],
            "probabilities": probabilities,
            "order": order_guess,
            "sizes": sizes,
        })

    return {"days": out_days}


def main() -> None:
    task = load_task()
    result = forecast(task)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main()
