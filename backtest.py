"""
Alpha Lab walk-forward backtest.

Strategies (pick with --strategy):
  trend   (default) time-series momentum across asset classes: each asset is long if its own
          12-1 month trend is up, short if down (Moskowitz, Ooi & Pedersen 2012)
  sector  cross-sectional 12-1 momentum across US sector ETFs (ranks sectors against each other)
Signals are computed from prices only; nothing is set by hand.
Alpha:      mu_i = sqrt(12) * IC * vol_i * z_i (annualized; IC is per monthly rebalance, 12 bets/yr)
Sizing:     w = kelly_fraction * Sigma^-1 mu   (fractional Kelly, default 0.15), gross exposure capped
Risk:       trailing 252-day covariance, shrunk toward its diagonal
Rebalance:  monthly, transaction costs charged on turnover, weights drift between rebalances
Checks:     strategy vs. SPY and equal-weight; 95% one-day VaR breach rate

Usage:  pip install yfinance pandas numpy matplotlib
        python backtest.py                    # trend strategy
        python backtest.py --strategy sector  # original sector version

Results are reported for the full period and split at --split (default 2013-01-01). The trend
paper was published in 2012, so 2013+ is a genuine out-of-sample test of the idea.
"""

import argparse
import numpy as np
import pandas as pd

SECTORS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU"]  # US sector ETFs, since 1998
# US stocks, intl stocks, EM stocks, 7-10y Treasuries, 20y+ Treasuries, gold, commodities, REITs
MULTI_ASSET = ["SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "VNQ"]  # common history from 2006
BENCH = "SPY"
REBALS_PER_YEAR = 12  # monthly rebalancing


def load_prices(tickers, start, end):
    import yfinance as yf
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return px.dropna(how="all").ffill().dropna()


def momentum_z(px_hist):
    """12-1 momentum: return from t-252 to t-21 trading days, z-scored across assets."""
    mom = px_hist.iloc[-21] / px_hist.iloc[-252] - 1
    return (mom - mom.mean()) / mom.std()


def trend_z(px_hist):
    """Time-series momentum: each asset's own 12-1 month return divided by its volatility
    (a trend t-stat), clipped to +/-3. Not compared across assets."""
    mom = px_hist.iloc[-21] / px_hist.iloc[-252] - 1
    vol = px_hist.iloc[-252:].pct_change().std() * np.sqrt(252)
    return (mom / vol).clip(-3, 3)


def shrunk_cov(rets, shrink=0.5):
    S = rets.cov().values * 252
    return (1 - shrink) * S + shrink * np.diag(np.diag(S))


def backtest(px, signal_fn, ic=0.05, kelly=0.15, gross_cap=2.0, cost_bps=10, lookback=252):
    rets = px.pct_change().dropna()
    month_ends = rets.groupby(rets.index.to_period("M")).tail(1).index
    rebal_dates = [d for d in month_ends if rets.index.get_loc(d) >= lookback]

    w = np.zeros(px.shape[1])
    daily, var_hits, var_days, turnover = [], 0, 0, []
    var95 = 0.0

    for i, day in enumerate(rets.index):
        if i > 0:
            r = rets.iloc[i].values
            port_r = float(w @ r)
            daily.append((day, port_r))
            if var95 > 0:
                var_days += 1
                var_hits += port_r < -var95
            # drift weights with prices
            w = w * (1 + r)
            w = w / (1 + port_r) if (1 + port_r) != 0 else w

        if day in rebal_dates:
            loc = rets.index.get_loc(day)
            hist_px = px.loc[:day]
            window = rets.iloc[loc - lookback + 1: loc + 1]   # data up to and including today only
            z = signal_fn(hist_px).values
            vol = window.std().values * np.sqrt(252)
            mu = np.sqrt(REBALS_PER_YEAR) * ic * vol * z   # annualized alpha
            Sigma = shrunk_cov(window)
            w_new = kelly * np.linalg.solve(Sigma, mu)
            gross = np.abs(w_new).sum()
            if gross > gross_cap:
                w_new *= gross_cap / gross
            to = np.abs(w_new - w).sum()
            turnover.append(to)
            if daily:  # charge costs on the rebalance day
                d, r_ = daily[-1]
                daily[-1] = (d, r_ - to * cost_bps / 1e4)
            w = w_new
            var95 = 1.645 * np.sqrt(w @ Sigma @ w / 252)

    s = pd.Series(dict(daily))
    s = s[s.index >= rebal_dates[0]]
    return s, var_hits / max(var_days, 1), float(np.mean(turnover)) if turnover else 0.0


def stats(r):
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    return {
        "Ann. return": eq.iloc[-1] ** (1 / yrs) - 1,
        "Ann. vol": r.std() * np.sqrt(252),
        "Sharpe": r.mean() / r.std() * np.sqrt(252),
        "Max drawdown": (eq / eq.cummax() - 1).min(),
    }


def show(title, strat, ew, spy):
    table = pd.DataFrame({"Strategy": stats(strat), "Equal-weight": stats(ew), "SPY": stats(spy)})
    print(f"\n{title}: {strat.index[0].date()} to {strat.index[-1].date()}")
    shown = table.copy().astype(object)
    for row in shown.index:
        fmt = "{:.2f}" if row == "Sharpe" else "{:.1%}"
        shown.loc[row] = [fmt.format(v) for v in table.loc[row]]
    print(shown.to_string())
    print(f"Correlation to SPY: {strat.corr(spy):.2f}")
    return table


def run(px, bench_px, args):
    signal_fn = trend_z if args.strategy == "trend" else momentum_z
    strat, breach, avg_to = backtest(px, signal_fn, args.ic, args.kelly, args.gross_cap, args.cost_bps)
    ew = px.pct_change().dropna().mean(axis=1).loc[strat.index]
    spy = bench_px.pct_change().dropna().reindex(strat.index).fillna(0)

    print(f"\nStrategy: {args.strategy}  |  universe: {', '.join(px.columns)}")
    table = show("Full period", strat, ew, spy)
    split = pd.Timestamp(args.split)
    for name, mask in [("Before split", strat.index < split), ("After split (out-of-sample)", strat.index >= split)]:
        if mask.sum() > 252:
            show(name, strat[mask], ew[mask], spy[mask])
    print()
    print(f"Avg monthly turnover:      {avg_to:.0%}")
    print(f"95% VaR breach rate:       {breach:.1%}  (target 5.0%)")
    table.to_csv("backtest_results.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, r in [("Strategy", strat), ("Equal-weight", ew), ("SPY", spy)]:
        ax.plot((1 + r).cumprod(), label=name)
    ax.set_title("Walk-forward backtest (growth of $1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("equity_curve.png", dpi=150)
    print("Saved backtest_results.csv and equity_curve.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["trend", "sector"], default="trend")
    p.add_argument("--start", default=None, help="default: 2006-03-01 (trend), 2014-01-01 (sector)")
    p.add_argument("--split", default="2013-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--ic", type=float, default=0.05)
    p.add_argument("--kelly", type=float, default=0.15)
    p.add_argument("--gross-cap", type=float, default=2.0)
    p.add_argument("--cost-bps", type=float, default=10)
    p.add_argument("--synthetic", action="store_true", help="run on random data (smoke test)")
    a = p.parse_args()
    universe = MULTI_ASSET if a.strategy == "trend" else SECTORS
    a.start = a.start or ("2006-03-01" if a.strategy == "trend" else "2014-01-01")

    if a.synthetic:
        rng = np.random.default_rng(0)
        idx = pd.bdate_range(a.start, a.end)
        cols = list(dict.fromkeys(universe + [BENCH]))
        r = rng.normal(0.0003, 0.012, (len(idx), len(cols)))
        allpx = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=cols)
    else:
        allpx = load_prices(list(dict.fromkeys(universe + [BENCH])), a.start, a.end)
    run(allpx[universe], allpx[BENCH], a)
