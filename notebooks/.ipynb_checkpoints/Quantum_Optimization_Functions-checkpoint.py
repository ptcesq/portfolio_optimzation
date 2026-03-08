
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import yfinance as yf
import numpy as np
from itertools import product

#------------------------------------------------------------------------#

def get_assets(
    companies_list = [], 
    start_date: str = "2024-07-01",
    end_date: str = "2025-06-30",
    sample_size: int = 10, 
    companies_file: Path = Path(
        r"../data/S_n_P_500.csv"
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

##########################################################################
#    Helper Functions 
##########################################################################

def compute_expected_return_vector(Returns_Matrix: np.ndarray) -> np.ndarray:
    """
    Compute Expected_Return_Vector using arithmetic mean daily returns
    scaled to annual (non-compounded).
    """
    Mean_Daily_Returns = Returns_Matrix.mean(axis=0)
    Expected_Return_Vector = 252 * Mean_Daily_Returns
    return Expected_Return_Vector

#--------------------------------------------------------------------------#

def build_Q_Matrix(Covariance_Matrix: np.ndarray,
                   Expected_Return_Vector: np.ndarray,
                   Risk_Aversion_Parameter: float) -> np.ndarray:
    """
    Build the Q_Matrix for a mean-variance portfolio QUBO with no constraints.

    E(x) = x^T Q_Matrix x
    where Q_Matrix = lambda * Covariance_Matrix - diag(Expected_Return_Vector)
    """
    N_Assets = Covariance_Matrix.shape[0]
    Diagonal_Return_Matrix = np.diag(Expected_Return_Vector)

    Q_Matrix = Risk_Aversion_Parameter * Covariance_Matrix - Diagonal_Return_Matrix

    assert Q_Matrix.shape == (N_Assets, N_Assets)
    return Q_Matrix

#-------------------------------------------------------------------------#

def generate_portfolios(assets: List[str]) -> List[Dict[str, Any]]:
    """
    Generate all non-empty portfolio combinations using a binary
    QUBO-style encoding.

    In a QUBO formulation of portfolio selection, each asset i
    is associated with a binary decision variable:

        x_i ∈ {0, 1}

    where:
        x_i = 1  → asset i is included in the portfolio
        x_i = 0  → asset i is excluded

    For n assets, there are 2^n possible binary configurations.
    This function enumerates all configurations EXCEPT the
    all-zero vector (the empty portfolio).

    Parameters
    ----------
    assets : List[str]
        A list of asset identifiers (e.g., ticker symbols).

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries. Each dictionary contains:
            - "binary_vector": List[int]
                  The QUBO decision vector.
            - "portfolio": List[str]
                  The selected assets corresponding to that vector.

        The total number of returned portfolios is (2^n - 1).
    """

    n = len(assets)
    portfolios = []

    # Generate all binary combinations of length n
    for binary_vector in product([0, 1], repeat=n):

        # Skip the empty portfolio (all zeros)
        if sum(binary_vector) == 0:
            continue

        selected_assets = [
            asset for asset, bit in zip(assets, binary_vector) if bit == 1
        ]

        portfolios.append({
            "binary_vector": list(binary_vector),
            "portfolio": selected_assets
        })

    return portfolios

#-------------------------------------------------------------------#

def compute_QUBO_Energy(Binary_Decision_Vector: np.ndarray,
                        Q_Matrix: np.ndarray) -> float:
    """
    Compute the QUBO energy E(x) = x^T Q_Matrix x for a given binary vector x.
    """
    return float(Binary_Decision_Vector.T @ Q_Matrix @ Binary_Decision_Vector)


#---------------------------------------------------------------------#

def compute_portfolio_risk_and_return(Binary_Decision_Vector: np.ndarray,
                                      Covariance_Matrix: np.ndarray,
                                      Expected_Return_Vector: np.ndarray):
    """
    Compute portfolio risk (variance) and expected return for a given binary vector.
    """
    Portfolio_Risk = float(Binary_Decision_Vector.T @ Covariance_Matrix @ Binary_Decision_Vector)
    Portfolio_Return = float(Expected_Return_Vector @ Binary_Decision_Vector)
    return Portfolio_Risk, Portfolio_Return

