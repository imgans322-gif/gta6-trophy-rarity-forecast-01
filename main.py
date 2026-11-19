#!/usr/bin/env python3
"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every case file (*.in) in $MAKEFUN_INPUT_DIR (default /input) and writes
one forecast per case to $MAKEFUN_OUTPUT_DIR (default /output): five quantiles
of each trophy's Exophase earn rate at the case's target instant.

Model, per trophy:
  * Collapse the read history into distinct Exophase snapshots (consecutive
    reads with identical players_tracked and rates are one page state).
  * Track two series over the snapshots: the tracked-player count N(t) and
    the trophy's earner count E(t) = rate(t) * N(t) / 100.
  * Extrapolate both on the log scale with damped least-squares slopes out
    to the target; the ratio gives an earners/players median forecast.
  * Blend that with a logit-rate trend fit against ln(1 + hours since
    release), the shape the earn-rate curve follows as players progress.
  * Spread: a logit-scale sigma combining a horizon random-walk term, the
    trophy's own snapshot-to-snapshot volatility, and the disagreement
    between the two median estimates.

Standard library only, deterministic, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
EPS_LO = 0.01
EPS_HI = 99.99


def parse_time(text):
    """Parse an ISO-8601 timestamp (with Z or offset) to epoch seconds."""
    if text is None:
        return None
    t = text.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def clamp_rate(x):
    return min(EPS_HI, max(EPS_LO, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def linfit(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b, residual_sd)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0 or n < 2:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / n)
    return a, b, sd


def snapshots(case):
    """Collapse history reads into distinct page states.

    Returns a list of (epoch_seconds, players_tracked, rates list), one per
    distinct (players, rates) state, keeping the FIRST time the state was
    seen (the refresh moment, to within one read interval).
    """
    out = []
    prev_key = None
    for read in case.get("history", []):
        rates = read.get("rates") or []
        players = read.get("players_tracked") or 0
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        t = parse_time(read.get("at") or read.get("slot"))
        if t is None or players <= 0:
            continue
        out.append((t, float(players), rates))
    return out


def forecast_case(case):
    trophies = case.get("trophies", [])
    at_t = parse_time(case.get("at"))
    target_t = parse_time(case.get("target"))
    if at_t is None or target_t is None:
        horizon_h = (case.get("horizon_minutes") or 0) / 60.0
        base = at_t or 0.0
        target_t = base + horizon_h * 3600.0
        at_t = base
    horizon_h = max((target_t - at_t) / 3600.0, 0.0)
    horizon_d = horizon_h / 24.0

    release_t = None
    game = case.get("game") or {}
    if game.get("release_date"):
        release_t = parse_time(str(game["release_date"])[:10] + "T00:00:00Z")
    snaps = snapshots(case)
    if not snaps:
        # No usable history: flat, wide, honest rows.
        rows = []
        for tr in trophies:
            rows.append({"id": tr["id"],
                         "q": [0.05, 0.3, 1.0, 3.0, 12.0]})
        return {"trophies": rows}
    if release_t is None:
        release_t = snaps[0][0]

    # Player-count trajectory on the log scale (damped extrapolation).
    ts = [s[0] for s in snaps]
    t0 = ts[0]
    hours = [(t - t0) / 3600.0 for t in ts]
    log_n = [math.log(s[1]) for s in snaps]
    _, slope_n, _ = linfit(hours, log_n)
    span_h = max(hours[-1], 0.0)
    damp_n = 0.55
    log_n_target = log_n[-1] + slope_n * horizon_h * damp_n
    log_n_target = min(log_n_target, log_n[-1] + math.log(25.0))
    log_n_target = max(log_n_target, log_n[-1] - math.log(4.0))
    n_target = math.exp(log_n_target)

    rows = []
    for idx, tr in enumerate(trophies):
        pts = []  # (hours_since_first, hours_since_release, earners, rate)
        for (t, players, rates), h in zip(snaps, hours):
            r = rates[idx] if idx < len(rates) else None
            if r is None:
                continue
            earners = r * players / 100.0
            hrs_rel = max((t - release_t) / 3600.0, 0.0)
            pts.append((h, hrs_rel, earners, r))
        if not pts:
            rows.append({"id": tr["id"], "q": [0.02, 0.1, 0.4, 1.5, 6.0]})
            continue
        last_rate = pts[-1][3]
        last_earners = pts[-1][2]

        medians = []
        weights = []
        if len(pts) >= 2:
            # Earners/players extrapolation.
            xs = [p[0] for p in pts]
            ys = [math.log(p[2] + 0.5) for p in pts]
            _, slope_e, _ = linfit(xs, ys)
            damp_e = 0.6
            log_e_target = ys[-1] + slope_e * horizon_h * damp_e
            log_e_target = min(log_e_target,
                               ys[-1] + math.log(400.0))
            e_target = max(math.exp(log_e_target) - 0.5, 0.0)
            rate_en = 100.0 * e_target / n_target if n_target > 0 else last_rate
            medians.append(logit(rate_en))
            weights.append(0.65)
            # Logit trend against ln(1 + hours since release).
            xs2 = [math.log1p(p[1]) for p in pts]
            ys2 = [logit(p[3]) for p in pts]
            a2, b2, _ = linfit(xs2, ys2)
            tgt_rel = max((target_t - release_t) / 3600.0, 0.0)
            rate_tr = inv_logit(a2 + b2 * math.log1p(tgt_rel))
            medians.append(logit(rate_tr))
            weights.append(0.35)
        if medians:
            wsum = sum(weights)
            med_logit = sum(m * w for m, w in zip(medians, weights)) / wsum
            disagree = 0.0
            if len(medians) == 2:
                disagree = abs(medians[0] - medians[1])
        else:
            med_logit = logit(last_rate)
            disagree = 0.0

        # Spread: horizon random walk + own volatility + model disagreement.
        sigma = 0.45 * math.sqrt(max(horizon_d, 0.01))
        if len(pts) >= 3:
            steps = []
            for i in range(1, len(pts)):
                dt_h = max(pts[i][0] - pts[i - 1][0], 0.05)
                dl = abs(logit(pts[i][3]) - logit(pts[i - 1][3]))
                steps.append(dl / math.sqrt(dt_h / 24.0))
            if steps:
                vol = sum(steps) / len(steps)
                sigma = math.sqrt(sigma ** 2 + (0.6 * vol) ** 2)
        sigma += 0.3 * disagree
        if len(snaps) <= 1:
            sigma = max(sigma, 1.1)
        sigma = min(max(sigma, 0.35), 1.8)

        qs = []
        for z in Z:
            q = inv_logit(med_logit + z * sigma)
            qs.append(round(min(100.0, max(0.0, q)), 4))
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
