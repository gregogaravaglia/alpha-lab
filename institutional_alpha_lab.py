import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtest import backtest, stats, trend_z, MULTI_ASSET, BENCH, REBALS_PER_YEAR

st.set_page_config(page_title="Institutional Alpha Lab", layout="wide")

st.title("Institutional Alpha Lab")
st.markdown("Fundamental Law + Mean-Variance + Fractional Kelly + Backtest + Monte Carlo")

# ======================================================
# Sidebar Inputs
# ======================================================

st.sidebar.header("Portfolio Inputs")

tickers_input = st.sidebar.text_input(
    "Tickers (comma separated)",
    value=",".join(MULTI_ASSET),
)
tickers = list(dict.fromkeys(t.strip().upper() for t in tickers_input.split(",") if t.strip()))

signal_source = st.sidebar.radio(
    "Signal source",
    ["Trend (computed from prices)", "Manual z-scores"],
    help="Trend: each asset's own 12-1 month return / volatility, clipped to ±3. "
         "Manual: your own views, set with the sliders below.",
)

ic = st.sidebar.slider(
    "Information Coefficient (IC, per monthly rebalance)", 0.0, 0.15, 0.05, 0.005,
    help="Correlation between the signal and the next month's return. 0.05 is a realistic level of skill.",
)
kelly_fraction = st.sidebar.slider(
    "Kelly Scaling", 0.05, 1.0, 0.15, 0.05,
    help="Fraction of the full-Kelly bet. Full Kelly maximizes long-run growth but is very volatile; "
         "professionals typically use a small fraction.",
)
gross_cap = st.sidebar.slider("Max gross exposure (x capital)", 0.5, 3.0, 2.0, 0.25)
capital = st.sidebar.number_input("Capital ($)", min_value=10000, value=1_000_000)

# ======================================================
# Pull Market Data
# ======================================================

@st.cache_data
def get_data(tickers):
    data = yf.download(
        list(dict.fromkeys(list(tickers) + [BENCH])),
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise ValueError("No data returned. Check ticker symbols.")
    return data["Close"]


try:
    all_px = get_data(tuple(tickers))
except ValueError as e:
    st.error(str(e))
    st.stop()

missing = [t for t in tickers if t not in all_px.columns or all_px[t].dropna().empty]
if missing:
    st.error(f"No data for: {', '.join(missing)}")
    st.stop()

prices = all_px[tickers].dropna()
bench_px = all_px[BENCH].reindex(prices.index).ffill()

if len(prices) < 300:
    st.error("Need at least ~300 trading days of common history for these tickers.")
    st.stop()

st.caption(f"Common price history: {prices.index[0].date()} to {prices.index[-1].date()}")

recent = prices.iloc[-504:]            # last 2 years for the current snapshot
returns = recent.pct_change().dropna()
annual_vol = returns.std() * np.sqrt(252)
cov_matrix = returns.cov() * 252
corr_matrix = returns.corr()

# ======================================================
# Signals -> Expected Returns (Fundamental Law alpha model)
# ======================================================

if signal_source.startswith("Trend"):
    z_scores = trend_z(prices).reindex(tickers)
    st.sidebar.markdown("### Current trend z-scores")
    st.sidebar.dataframe(z_scores.rename("z").to_frame().style.format("{:+.2f}"))
else:
    st.sidebar.markdown("### Signal Z-Scores")
    z_scores = pd.Series(
        {t: st.sidebar.slider(f"{t} Signal", -3.0, 3.0, 1.0, 0.1) for t in tickers}
    )

# IC applies to each monthly bet; 12 independent bets a year scale annual alpha by sqrt(12)
mu = (np.sqrt(REBALS_PER_YEAR) * ic * annual_vol * z_scores).reindex(tickers).values
Sigma = cov_matrix.values

# ======================================================
# Optimal Allocation (fractional Kelly on mean-variance weights)
# ======================================================

try:
    weights = kelly_fraction * np.linalg.solve(Sigma, mu)
except np.linalg.LinAlgError:
    st.error("Covariance matrix is singular (duplicate or perfectly correlated tickers?).")
    st.stop()

gross = np.abs(weights).sum()
if gross > gross_cap:
    weights *= gross_cap / gross       # cap leverage; otherwise Kelly scaling sets the size

portfolio_expected_return = weights @ mu
portfolio_vol = np.sqrt(weights @ Sigma @ weights)
portfolio_sharpe = portfolio_expected_return / portfolio_vol if portfolio_vol > 0 else 0.0
breadth = len(tickers) * REBALS_PER_YEAR      # independent bets per year
fundamental_law_sr = ic * np.sqrt(breadth)

# ======================================================
# Diversification Attribution
# ======================================================

full_var = weights @ Sigma @ weights
diag_var = weights @ np.diag(np.diag(Sigma)) @ weights
div_benefit = diag_var - full_var

# ======================================================
# Monte Carlo Simulation
# ======================================================

def monte_carlo_simulation(mu, Sigma, weights, days=252, simulations=5000, seed=42):
    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(mu / 252, Sigma / 252, size=(simulations, days))
    port_daily = sims @ weights
    return np.prod(1 + port_daily, axis=1) - 1

mc_results = monte_carlo_simulation(mu, Sigma, weights)
mc_mean = mc_results.mean()
mc_prob_loss = np.mean(mc_results < 0)
mc_var_5 = np.percentile(mc_results, 5)

# ======================================================
# DISPLAY: current portfolio
# ======================================================

st.subheader("Asset Allocations (today)")

allocation_df = pd.DataFrame({
    "Ticker": tickers,
    "Signal z": z_scores.reindex(tickers).values,
    "Expected Return": mu,
    "Weight": weights,
    "Position": ["LONG" if w > 0 else "SHORT" if w < 0 else "FLAT" for w in weights],
    "Capital Allocation ($)": weights * capital,
})
st.dataframe(allocation_df.style.format({
    "Signal z": "{:+.2f}",
    "Expected Return": "{:.2%}",
    "Weight": "{:.2%}",
    "Capital Allocation ($)": "${:,.0f}",
}))
st.caption(f"Gross exposure: {np.abs(weights).sum():.2f}x capital  |  Net: {weights.sum():+.2f}x")

st.subheader("Portfolio Metrics (model)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected Return (alpha)", f"{portfolio_expected_return:.2%}",
          help="Annual return forecast from the signal alone, at the current Kelly scaling. "
               "Excludes the normal return from simply holding the assets.")
c2.metric("Volatility", f"{portfolio_vol:.2%}")
c3.metric("Ex-ante Sharpe", f"{portfolio_sharpe:.2f}")
c4.metric("Fundamental Law Sharpe (IC×√Breadth)", f"{fundamental_law_sr:.2f}",
          help=f"Breadth = {len(tickers)} assets × {REBALS_PER_YEAR} rebalances = {breadth} bets a year")
st.metric("Diversification Benefit (Variance Reduction)", f"{div_benefit:.4f}")

st.subheader("Correlation Matrix (last 2 years)")
st.dataframe(corr_matrix.style.format("{:.2f}"))

# ======================================================
# DISPLAY: walk-forward backtest
# ======================================================

st.subheader("Walk-Forward Backtest")

if signal_source.startswith("Trend"):
    cost_bps = st.slider("Transaction cost (bps per unit of turnover)", 0, 50, 10)
    with st.spinner("Running backtest..."):
        strat, breach, avg_to = backtest(
            prices, trend_z, ic=ic, kelly=kelly_fraction, gross_cap=gross_cap, cost_bps=cost_bps
        )
    ew = prices.pct_change().dropna().mean(axis=1).reindex(strat.index)
    spy = bench_px.pct_change().reindex(strat.index).fillna(0)

    table = pd.DataFrame({"Strategy": stats(strat), "Equal-weight": stats(ew), "SPY": stats(spy)})
    shown = table.copy().astype(object)
    for row in shown.index:
        fmt = "{:.2f}" if row == "Sharpe" else "{:.1%}"
        shown.loc[row] = [fmt.format(v) for v in table.loc[row]]
    st.caption(f"{strat.index[0].date()} to {strat.index[-1].date()}  |  monthly rebalance, "
               f"signals use only data available at each date")
    st.dataframe(shown)

    b1, b2, b3 = st.columns(3)
    b1.metric("Correlation to SPY", f"{strat.corr(spy):.2f}")
    b2.metric("Avg monthly turnover", f"{avg_to:.0%}")
    b3.metric("95% VaR breach rate (target 5%)", f"{breach:.1%}")

    curves = pd.DataFrame({
        "Strategy": (1 + strat).cumprod(),
        "Equal-weight": (1 + ew).cumprod(),
        "SPY": (1 + spy).cumprod(),
    })
    st.line_chart(curves)
else:
    st.info("Backtest is only available with trend signals. Manual z-scores are today's views; "
            "replaying them over past data would use hindsight and the result would be meaningless.")

# ======================================================
# DISPLAY: Monte Carlo
# ======================================================

st.subheader("Monte Carlo Simulation (1-Year, current portfolio)")
m1, m2, m3 = st.columns(3)
m1.metric("Expected 1Y Return", f"{mc_mean:.2%}")
m2.metric("Probability of Loss", f"{mc_prob_loss:.2%}")
m3.metric("5% Worst Case (VaR)", f"{mc_var_5:.2%}")

fig, ax = plt.subplots()
ax.hist(mc_results, bins=50)
ax.set_title("Monte Carlo Distribution of 1Y Returns")
st.pyplot(fig)

st.warning("Model assumes stable IC, normally distributed returns and constant covariance. "
           "Normal VaR understates tail risk (backtest breach rate is above 5%).")
