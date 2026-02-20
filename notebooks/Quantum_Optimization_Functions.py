
from pathlib import Path
from typing import List
import pandas as pd
import yfinance as yf
import numpy as np

def get_assets(
    companies_list = [], 
    start_date: str = "2024-07-01",
    end_date: str = "2025-06-30",
    sample_size: int = 10, 
    companies_file: Path = Path(
        r"C:\Users\patri\OneDrive\Projects\pyton\portfolio_optimzation\data\S_n_P_500.csv"
    )    
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieve historical price data and compute portfolio statistics.

    If a list of tickers is provided, those assets are used.
    Otherwise, a random sample of tickers is selected from the
    provided companies file.

    Parameters
    ----------
    companies_list : list[str], optional
        List of ticker symbols to analyze. If empty, a random
        sample is drawn from `companies_file`.

    start_date : str
        Start date for historical price data (YYYY-MM-DD).

    end_date : str
        End date for historical price data (YYYY-MM-DD).

    sample_size : int
        Number of random tickers to select if `companies_list`
        is empty.

    companies_file : Path
        CSV file containing available tickers (first column assumed).

    Returns
    -------
    summary_df : pandas.DataFrame
        Summary statistics per asset containing:
            - Ticker
            - Last Close price
            - CAGR (%) — compound annual growth rate
            - Volatility (%) — annualized standard deviation

    daily_returns : pandas.DataFrame
        DataFrame of daily percentage returns for all assets.
        Used for covariance and correlation matrix calculations.

    Financial Assumptions
    ---------------------
    - 252 trading days per year
    - CAGR computed using geometric compounding
    - Volatility annualized via sqrt(252)
    """

    # ---------------------------------------------------------------
    # Determine ticker domain
    # ---------------------------------------------------------------
    tickers: List[str] = companies_list

    if len(tickers) < 1:
        # If no tickers provided, randomly sample from file
        companies = pd.read_csv(companies_file)
        tickers = companies.iloc[:, 0].sample(
            n=sample_size, replace=False
        ).tolist()

    # ---------------------------------------------------------------
    # Download daily price data from Yahoo Finance
    # ---------------------------------------------------------------
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False
    )

    trading_days = 252
    rows = []

    # ---------------------------------------------------------------
    # Construct Close price DataFrame across all assets
    # ---------------------------------------------------------------
    close_df = pd.DataFrame()

    for ticker in tickers:
        try:
            close_df[ticker] = data[ticker]["Close"]
        except Exception:
            close_df[ticker] = None

    # Remove rows with missing price data
    close_df = close_df.dropna()

    # ---------------------------------------------------------------
    # Compute daily returns (used for risk modeling)
    # ---------------------------------------------------------------
    daily_returns = close_df.pct_change().dropna()

    # ---------------------------------------------------------------
    # Compute per-asset summary statistics
    # ---------------------------------------------------------------
    for ticker in tickers:
        try:
            close_prices = close_df[ticker].dropna()
            last_close = round(close_prices.iloc[-1], 2)

            returns = daily_returns[ticker]
            n_days = len(returns)

            # CAGR (geometric annualized return)
            annual_return = round(
                ((close_prices.iloc[-1] / close_prices.iloc[0]) 
                 ** (trading_days / n_days) - 1) * 100,
                6
            )

            # Annualized volatility (std * sqrt(252))
            annual_volatility = round(
                returns.std() * (trading_days ** 0.5) * 100,
                6
            )

            rows.append([
                ticker,
                last_close,
                annual_return,
                annual_volatility
            ])

        except Exception:
            rows.append([ticker, None, None, None])

    summary_df = pd.DataFrame(
        rows,
        columns=["Ticker", "Last Close", "CAGR (%)", "Volatility (%)"]
    )

    return summary_df, daily_returns


# --- Helper: QUBO and Ising energy evaluation ---

def qubo_value(bits, Q, c):
    """Evaluate QUBO value for given bitstring."""
    x = np.array(bits)
    return float(x @ Q @ x + c @ x)

def ising_value(bits, h, J):
    """Evaluate Ising energy for given bitstring."""
    z = 1 - 2 * np.array(bits)  # Binary-to-spin conversion
    e = np.sum(h * z)
    for i in range(len(bits)):
        for j in range(i+1, len(bits)):
            e += J[i, j] * z[i] * z[j]
    return float(e)

# --- Piecewise budget penalty ---
def budget_piecewise_penalty(cost, band_low, band_high, alpha, budget):
    if cost < band_low:
        return alpha * (band_low - cost) ** 2 / budget**2
    elif cost > band_high:
        return alpha * (cost - band_high) ** 2 / budget**2
    return 0.0

