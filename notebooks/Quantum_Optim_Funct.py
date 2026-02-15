
from pathlib import Path
from typing import List
import pandas as pd
import yfinance as yf


def get_random_assets(
    asset_sample_size: int = 10,
    start_date: str = "2024-07-01",
    end_date: str = "2025-06-30",
    companies_file: Path = Path(
        r"C:\Users\patri\OneDrive\Projects\pyton\portfolio_optimzation\data\S_n_P_500.csv"
    )
) -> pd.DataFrame:
    """
    Return summary statistics for a random sample of S&P 500 assets.

    Parameters
    ----------
    asset_sample_size : int
        Number of tickers to randomly select.
    start_date : str
        Start date for historical data (YYYY-MM-DD).
    end_date : str
        End date for historical data (YYYY-MM-DD).
    companies_file : Path
        Path to the CSV file containing the list of S&P 500 tickers.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        - Ticker
        - Last Close
        - Total Return
        - Daily Volatility
    """

    # --- Load ticker list ----------------------------------------------------
    companies = pd.read_csv(companies_file)
    tickers: List[str] = companies.iloc[:, 0].sample(
        n=asset_sample_size, replace=False
    ).tolist()

    # --- Download price data -------------------------------------------------
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False
    )

    # --- Compute metrics -----------------------------------------------------
    rows = []

    for ticker in tickers:
        try:
            close_prices = data[ticker]["Close"].dropna()

            if close_prices.empty:
                raise ValueError("No price data")

            last_close = close_prices.iloc[-1]
            total_return = (last_close / close_prices.iloc[0]) - 1
            daily_volatility = close_prices.pct_change().dropna().std()

            rows.append([ticker, last_close, total_return, daily_volatility])

        except Exception:
            rows.append([ticker, None, None, None])

    # --- Build output DataFrame ----------------------------------------------
    return pd.DataFrame(
        rows,
        columns=["Ticker", "Last Close", "Total Return", "Daily Volatility"]
    )


def get_assets(
    companies_list = [], 
    start_date: str = "2024-07-01",
    end_date: str = "2025-06-30"    
) -> pd.DataFrame:
    """
    Return summary statistics for a list of assets.

    Parameters
    ----------
    companies_list: Array 
        An array of the tickers to be fetched.
    start_date : str
        Start date for historical data (YYYY-MM-DD).
    end_date : str
        End date for historical data (YYYY-MM-DD).


    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        - Ticker
        - Last Close
        - Total Return
        - Daily Volatility
    """

    # --- Load ticker list ----------------------------------------------------
    tickers: List[str] = companies_list
    if len(tickers) < 1: 
        return None 

    # --- Download price data -------------------------------------------------
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False
    )

    # --- Compute metrics -----------------------------------------------------
    rows = []

    for ticker in tickers:
        try:
            close_prices = data[ticker]["Close"].dropna()

            if close_prices.empty:
                raise ValueError("No price data")

            last_close = close_prices.iloc[-1]
            total_return = (last_close / close_prices.iloc[0]) - 1
            daily_volatility = close_prices.pct_change().dropna().std()

            rows.append([ticker, last_close, total_return, daily_volatility])

        except Exception:
            rows.append([ticker, None, None, None])

    # --- Build output DataFrame ----------------------------------------------
    return pd.DataFrame(
        rows,
        columns=["Ticker", "Last Close", "Total Return", "Daily Volatility"]
    )
