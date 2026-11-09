# xstocks-forecast

[![make.fun · xstocks-movers-forecast-01](https://make.fun/badge/xstocks-movers-forecast-01.svg)](https://make.fun/bounty/xstocks-movers-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=xstocks-movers-forecast-01)

A forecaster for the make.fun bounty "Predict the next 5 trading days' moves of 9 Nasdaq
stocks": it reads the hourly on-chain price history of nine xStocks (tokenized Nasdaq
shares on Solana) and writes, for each of the five session days after entries close, a
probability for every name at every place of the day's order by return, a point guess at
that order, and a probability for every name's return in half-point bins from -10% to +10%.

Python 3.13, standard library only, deterministic (a seeded Monte Carlo), about five
seconds on one CPU.

## The model

- **Fat-tailed daily returns.** Each name's return is Student-t with 4 degrees of freedom,
  scaled to a forecast daily vol. Four degrees of freedom cover both the fat tails of
  daily stock returns and the uncertainty in the vol itself; a small inflation of every vol
  is applied because the log score punishes a vol set too low harder than one set too high.
- **Vol = regime prior, nudged by the history.** The history holds one session of hourly
  bars per name, so the realized vol it gives is noisy. Each name gets a prior daily vol
  for the week's regime (a high-vol tape after the Nasdaq's 5% drop on 2026-11-05), and the
  realized vol of the bars moves it in log space with a weight that decays with the
  horizon. Scheduled releases (CPI, PPI) widen their day; a Monday spans the weekend; the
  vol eases slightly for later days.
- **The first day is partly known.** The first window day's return runs from the previous
  session's close, and the history ends two hours after it. That post-close move is carried
  into the day's mean (shrunk a little, since thin after-hours pool prices partly revert),
  and the remaining variance is reduced accordingly. The close itself is estimated as the
  30-minute TWAP from the hour's open and close prices (a Brownian-bridge weighting).
- **One market factor, common tail.** Names co-move through a single factor inside a
  multivariate Student-t, so a wild day is wild for everyone. Place probabilities come from
  100,000 seeded draws of that joint model; bin probabilities are the exact marginal t
  probabilities, so sizes and order are two views of one distribution.
- **No zero anywhere.** Place probabilities carry a 1% uniform mix and bins a small floor,
  so no outcome is ever scored at the floor.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

In the grader the two variables are unset and the program reads `/input` and writes
`/output/forecast.json`.

`tools/score_local.py <forecast.json> <answers.json>` scores a forecast with the bounty's
rule (log score floored at 1e-4), given a JSON of `{date: {symbol: return}}`.

## Author

Rizal ([x.com/peper1zal](https://x.com/peper1zal)). MIT license.
