"""
Monte Carlo Derivatives Pricing Engine — SPY Calibrated
============================================================
Simulates Geometric Brownian Motion under the risk-neutral measure to
price options via Monte Carlo simulation.

Features:
  1. European call pricing with a 95% confidence interval (quantifies
     estimator uncertainty, not just a single point estimate)
  2. Validation against the closed-form Black-Scholes formula
  3. Asian (arithmetic-average) call pricing — a path-dependent payoff
     with NO closed-form solution, which is the real reason trading
     desks reach for Monte Carlo over Black-Scholes
  4. Calibration to real market data:
       - Volatility estimated from actual SPY historical prices
       - Compared against SPY's real market-implied volatility to
         illustrate the historical-vs-implied volatility gap (the
         "volatility risk premium")

Data sources (see bottom of file for full citations):
  - Historical prices: user-supplied SPY daily OHLCV CSV
  - Implied volatility, risk-free rate: pulled from public market data,
    dated August 2026
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================
# PART 1 — Calibrate to real market data
# ============================================================
def realized_volatility(csv_path, window=252):
    """Annualized volatility from real historical daily closes."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    recent = df.tail(window)
    daily_std = recent["log_return"].std()
    annual_vol = daily_std * np.sqrt(252)  # 252 = trading days per year

    return annual_vol, df["Close"].iloc[-1], df["Date"].iloc[-1].date()


sigma_historical, S0, price_date = realized_volatility(
    "spy_us_d.csv", window=252  # place this CSV in the same folder as this script
)

# Real market inputs (see citations at bottom of file)
sigma_implied = 0.126     # SPY 30-day ATM implied volatility, Aug 21 2026
r = 0.038                 # ~3-month T-bill yield, Aug 2026 (risk-free proxy)
K = round(S0)              # at-the-money strike
T = 30 / 365               # 30-day option, matching the implied-vol tenor


# ============================================================
# PART 2 — Monte Carlo European call, with confidence interval
# ============================================================
def monte_carlo_european_call(S0, K, r, sigma, T, M=200_000, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(M)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)
    discounted = np.exp(-r * T) * payoffs

    price = discounted.mean()
    se = discounted.std(ddof=1) / np.sqrt(M)          # standard error of the mean
    ci = (price - 1.96 * se, price + 1.96 * se)        # 95% confidence interval
    return price, se, ci


def black_scholes_call(S0, K, r, sigma, T):
    """Closed-form solution — our 'answer key' for validation."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def convergence_study(S0, K, r, sigma, T, path_counts, seed=42):
    """
    Runs ONE growing Monte Carlo simulation and checkpoints the price
    estimate and standard error at increasing path counts. This shows
    how a single simulation's estimate refines as it's allowed to run
    longer, rather than comparing unrelated independent simulations.
    """
    rng = np.random.default_rng(seed)
    M_max = max(path_counts)
    Z_all = rng.standard_normal(M_max)
    ST_all = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_all)
    discounted_all = np.exp(-r * T) * np.maximum(ST_all - K, 0)

    prices, ses = [], []
    for M in path_counts:
        subset = discounted_all[:M]
        prices.append(subset.mean())
        ses.append(subset.std(ddof=1) / np.sqrt(M))
    return np.array(prices), np.array(ses)


# ============================================================
# PART 3 — Monte Carlo Asian (arithmetic-average) call
# ============================================================
def monte_carlo_asian_call(S0, K, r, sigma, T, M=200_000, N=30, seed=42):
    """
    Path-dependent option: payoff depends on the AVERAGE price over the
    option's life, not just the final price. Black-Scholes cannot price
    this (the average of lognormal prices is not itself lognormal), so
    this is a case where Monte Carlo is doing something a formula can't.

    N = number of steps simulated per path (here, one per day).
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    Z = rng.standard_normal((M, N))                      # M paths x N steps
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    paths = S0 * np.exp(log_paths)                        # simulated price at each step

    avg_price = paths.mean(axis=1)                        # average price per path
    payoffs = np.maximum(avg_price - K, 0)
    discounted = np.exp(-r * T) * payoffs

    price = discounted.mean()
    se = discounted.std(ddof=1) / np.sqrt(M)
    ci = (price - 1.96 * se, price + 1.96 * se)
    return price, se, ci, paths


# ============================================================
# RUN EVERYTHING
# ============================================================
if __name__ == "__main__":
    M = 200_000

    print("=" * 62)
    print("REAL MARKET CALIBRATION (from your SPY data + live market data)")
    print("=" * 62)
    print(f"SPY price ({price_date}):          ${S0:.2f}")
    print(f"Strike (at-the-money):             ${K}")
    print(f"Time to expiration:                30 days")
    print(f"Risk-free rate:                    {r*100:.2f}%")
    print(f"Historical volatility (1yr, real): {sigma_historical*100:.2f}%")
    print(f"Market-implied volatility (real):  {sigma_implied*100:.2f}%")
    print()

    print("=" * 62)
    print("1) EUROPEAN CALL — priced with HISTORICAL volatility")
    print("=" * 62)
    mc_price, mc_se, mc_ci = monte_carlo_european_call(S0, K, r, sigma_historical, T, M)
    bs_price = black_scholes_call(S0, K, r, sigma_historical, T)
    print(f"Monte Carlo price ({M:,} paths):    ${mc_price:.4f}")
    print(f"  95% confidence interval:          (${mc_ci[0]:.4f}, ${mc_ci[1]:.4f})")
    print(f"  Standard error:                   ${mc_se:.4f}")
    print(f"Black-Scholes price (exact):        ${bs_price:.4f}")
    inside = mc_ci[0] <= bs_price <= mc_ci[1]
    print(f"  -> Black-Scholes falls inside our 95% CI: {inside}")
    print()

    print("=" * 62)
    print("2) EUROPEAN CALL — priced with MARKET-IMPLIED volatility")
    print("=" * 62)
    bs_price_implied = black_scholes_call(S0, K, r, sigma_implied, T)
    gap = bs_price_implied - bs_price
    print(f"Black-Scholes price (implied vol):  ${bs_price_implied:.4f}")
    print(f"Gap vs. historical-vol price:        ${gap:+.4f}  ({gap/bs_price*100:+.1f}%)")
    print("Interpretation: the market's forward-looking volatility estimate")
    print("(implied vol) differs from what SPY actually realized historically.")
    print("Over long periods implied vol tends to average ABOVE realized vol")
    print("(the 'volatility risk premium') as compensation for option sellers'")
    print("risk-bearing — but on any single day, like here, the gap can be")
    print("small or even flip sign, which is exactly what we see.")
    print()

    print("=" * 62)
    print("3) ASIAN CALL (arithmetic average) — no closed-form solution exists")
    print("=" * 62)
    asian_price, asian_se, asian_ci, paths = monte_carlo_asian_call(
        S0, K, r, sigma_historical, T, M, N=30
    )
    print(f"Monte Carlo Asian call price:       ${asian_price:.4f}")
    print(f"  95% confidence interval:          (${asian_ci[0]:.4f}, ${asian_ci[1]:.4f})")
    print(f"European call price (same inputs):  ${mc_price:.4f}")
    print(f"  -> Asian option is cheaper by:    ${mc_price - asian_price:.4f}")
    print("Makes sense: averaging the price path smooths out extreme moves,")
    print("so the Asian payoff has lower variance -> lower option value.")
    print()

    print("=" * 62)
    print("4) CONVERGENCE ANALYSIS — how many simulations do we actually need?")
    print("=" * 62)
    path_counts = np.array([10, 30, 100, 300, 1000, 3000, 10_000, 30_000,
                             100_000, 300_000, 1_000_000])
    conv_prices, conv_ses = convergence_study(S0, K, r, sigma_historical, T, path_counts)
    print(f"{'Paths':>10} | {'Price':>10} | {'Std Error':>10} | {'Error vs BS':>12}")
    print("-" * 50)
    for M, p, se in zip(path_counts, conv_prices, conv_ses):
        print(f"{M:>10,} | ${p:>9.4f} | ${se:>9.4f} | ${abs(p - bs_price):>11.4f}")
    print()
    print("Notice the standard error shrinks roughly by 10x every time paths")
    print("increase 100x — that's the sqrt(N) relationship: to cut error in")
    print("half, you need 4x the simulations, not 2x. This is Monte Carlo's")
    print("central computational tradeoff.")

    # ============================================================
    # PART 4 — Save a visualization
    # ============================================================
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, works without a display
    import matplotlib.pyplot as plt

    sample = paths[:100]  # plot a readable subset of the 200,000 simulated paths

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for i in range(len(sample)):
        ax.plot(sample[i], linewidth=0.6, alpha=0.5, color="#2563eb")
    ax.plot(sample.mean(axis=0), linewidth=2.5, color="#111827", label="Mean path")
    ax.set_title(f"{len(sample)} Simulated SPY Price Paths (30 days)\ncalibrated to real historical volatility")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Simulated SPY price ($)")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    avg_prices = sample.mean(axis=1)
    ax.hist(avg_prices, bins=30, color="#2563eb", alpha=0.75, edgecolor="white")
    ax.axvline(K, color="#dc2626", linestyle="--", linewidth=2, label=f"Strike (K = ${K})")
    ax.set_title("Distribution of Path-Average Price\n(this drives the Asian option's payoff)")
    ax.set_xlabel("Average simulated price over the 30 days ($)")
    ax.set_ylabel("Number of simulated paths")
    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("spy_monte_carlo_chart.png", dpi=150, bbox_inches="tight")
    print()
    print("Saved chart: spy_monte_carlo_chart.png")

    # ============================================================
    # PART 5 — Convergence chart
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ci_low = conv_prices - 1.96 * conv_ses
    ci_high = conv_prices + 1.96 * conv_ses
    ax.plot(path_counts, conv_prices, "o-", color="#2563eb", label="Monte Carlo estimate")
    ax.fill_between(path_counts, ci_low, ci_high, color="#2563eb", alpha=0.2, label="95% CI")
    ax.axhline(bs_price, color="#dc2626", linestyle="--", linewidth=2, label="Black-Scholes (exact)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths (log scale)")
    ax.set_ylabel("Option price estimate ($)")
    ax.set_title("Price Estimate Converging to the Exact Answer")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(path_counts, conv_ses, "o-", color="#2563eb", label="Actual standard error")
    theoretical = conv_ses[0] * np.sqrt(path_counts[0] / path_counts)
    ax.plot(path_counts, theoretical, "--", color="#dc2626", label=r"Theoretical $1/\sqrt{N}$ slope")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of simulated paths (log scale)")
    ax.set_ylabel("Standard error ($, log scale)")
    ax.set_title("Error Shrinks with $1/\\sqrt{N}$ — Monte Carlo's Core Tradeoff")
    ax.legend()
    ax.grid(alpha=0.25, which="both")

    plt.tight_layout()
    plt.savefig("convergence_chart.png", dpi=150, bbox_inches="tight")
    print("Saved chart: convergence_chart.png")


# ============================================================
# DATA SOURCE CITATIONS
# ============================================================
# - SPY historical daily OHLCV: user-provided CSV (Yahoo Finance export)
# - SPY 30-day at-the-money implied volatility (12.6%) and 30-day
#   realized volatility context: OptiView (opti-view.com), data as of
#   Aug 21, 2026, OPRA data ~15 min delayed
# - Risk-free rate proxy (~3.8%): 3-month U.S. Treasury bill yield,
#   CNBC / FRED (DGS3MO), approx. as of late July/Aug 2026
