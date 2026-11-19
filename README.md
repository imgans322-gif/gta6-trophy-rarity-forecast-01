# GTA VI Trophy Rarity Forecast

![Competition badge](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)
![Rank badge](https://make.fun/badge/gta6-trophy-rarity-forecast-01/imgans322-gif/gta6-trophy-rarity-forecast-01/rank.svg)
![Reward badge](https://make.fun/badge/gta6-trophy-rarity-forecast-01/imgans322-gif/gta6-trophy-rarity-forecast-01/reward.svg)

Entry for the make.fun bounty [gta6-trophy-rarity-forecast-01](https://make.fun/bounties/gta6-trophy-rarity-forecast-01): forecast, for every Grand Theft Auto VI PlayStation 5 trophy, its Exophase earn rate at 2026-11-26T00:00:00Z as five quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

## How it works

`main.py` (Python 3.13, standard library only, deterministic, offline) reads each case file in `$MAKEFUN_INPUT_DIR` and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR`. Per trophy it:

- collapses the read history into distinct Exophase page snapshots (consecutive identical reads are one state),
- tracks the tracked-player count N(t) and the trophy's earner count E(t) = rate x N / 100,
- extrapolates both series on the log scale with damped least-squares slopes to the target and takes the ratio as one median estimate,
- blends that with a logit-rate trend fit against ln(1 + hours since release),
- sets the spread from a horizon random-walk term, the trophy's own snapshot-to-snapshot volatility, and the disagreement between the two median estimates.
