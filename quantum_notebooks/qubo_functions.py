import itertools
import numpy as np
import pandas as pd


def compute_expected_return_vector(
    Asset_Returns_DataFrame: pd.DataFrame,
) -> np.ndarray:
    """
    Compute expected (mean) returns for each asset from a DataFrame of returns.

    Parameters
    ----------
    Asset_Returns_DataFrame : pd.DataFrame
        Rows are time periods, columns are assets.

    Returns
    -------
    np.ndarray
        1D array of expected returns per asset, in column order.
    """
    Expected_Return_Vector = Asset_Returns_DataFrame.mean(axis=0).values
    return Expected_Return_Vector


def compute_covariance_matrix(
    Asset_Returns_DataFrame: pd.DataFrame,
) -> np.ndarray:
    """
    Compute covariance matrix of asset returns.

    Parameters
    ----------
    Asset_Returns_DataFrame : pd.DataFrame

    Returns
    -------
    np.ndarray
        2D covariance matrix.
    """
    Sigma_Covariance_Matrix = Asset_Returns_DataFrame.cov().values
    return Sigma_Covariance_Matrix


def build_qubo_matrix_mean_variance(
    Sigma_Covariance_Matrix: np.ndarray,
    Expected_Return_Vector: np.ndarray,
    Risk_Aversion_Parameter: float,
) -> np.ndarray:
    """
    Build QUBO matrix for unconstrained mean-variance portfolio selection.

    Q = lambda * Sigma - diag(mu)

    Parameters
    ----------
    Sigma_Covariance_Matrix : np.ndarray
        Covariance matrix of asset returns.
    Expected_Return_Vector : np.ndarray
        Expected returns per asset.
    Risk_Aversion_Parameter : float
        Trade-off parameter between risk and return.

    Returns
    -------
    np.ndarray
        QUBO matrix Q.
    """
    Risk_Term = Risk_Aversion_Parameter * Sigma_Covariance_Matrix
    Return_Term = np.diag(Expected_Return_Vector)
    Q_Matrix = Risk_Term - Return_Term
    return Q_Matrix


def generate_all_binary_portfolios(
    Number_of_Assets: int,
) -> list[tuple[int, ...]]:
    """
    Generate all binary portfolios for a given number of assets.

    Parameters
    ----------
    Number_of_Assets : int

    Returns
    -------
    list of tuples
        Each tuple is a binary vector (0/1) of length Number_of_Assets.
    """
    Binary_Portfolios = list(itertools.product([0, 1], repeat=Number_of_Assets))
    return Binary_Portfolios
 


def compute_portfolio_energy(
    Binary_Portfolio_Vector: np.ndarray,
    Q_Matrix: np.ndarray,
) -> float:
    """
    Compute QUBO energy x^T Q x for a given binary portfolio.

    Parameters
    ----------
    Binary_Portfolio_Vector : np.ndarray
        1D binary vector.
    Q_Matrix : np.ndarray
        QUBO matrix.

    Returns
    -------
    float
        Energy value.
    """
    Energy_Value = float(Binary_Portfolio_Vector.T @ Q_Matrix @ Binary_Portfolio_Vector)
    return Energy_Value


def compute_portfolio_return(
    Binary_Portfolio_Vector: np.ndarray,
    Expected_Return_Vector: np.ndarray,
) -> float:
    """
    Compute portfolio expected return for a binary portfolio.

    Parameters
    ----------
    Binary_Portfolio_Vector : np.ndarray
    Expected_Return_Vector : np.ndarray

    Returns
    -------
    float
        Portfolio expected return.
    """
    Portfolio_Return = float(Binary_Portfolio_Vector @ Expected_Return_Vector)
    return Portfolio_Return


def compute_portfolio_risk(
    Binary_Portfolio_Vector: np.ndarray,
    Sigma_Covariance_Matrix: np.ndarray,
) -> float:
    """
    Compute portfolio risk (variance) for a binary portfolio.

    Parameters
    ----------
    Binary_Portfolio_Vector : np.ndarray
    Sigma_Covariance_Matrix : np.ndarray

    Returns
    -------
    float
        Portfolio variance.
    """
    Portfolio_Risk = float(
        Binary_Portfolio_Vector.T @ Sigma_Covariance_Matrix @ Binary_Portfolio_Vector
    )
    return Portfolio_Risk


def evaluate_all_binary_portfolios(
    Binary_Portfolios: list[tuple[int, ...]],
    Q_Matrix: np.ndarray,
    Expected_Return_Vector: np.ndarray,
    Sigma_Covariance_Matrix: np.ndarray,
    Asset_Names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Evaluate energy, risk, and return for all binary portfolios.

    Parameters
    ----------
    Binary_Portfolios : list of tuples
    Q_Matrix : np.ndarray
    Expected_Return_Vector : np.ndarray
    Sigma_Covariance_Matrix : np.ndarray
    Asset_Names : list of str, optional
        Names of assets, used to label columns.

    Returns
    -------
    pd.DataFrame
        Columns: asset bits, Energy, Portfolio_Return, Portfolio_Risk.
    """
    Evaluation_Records = []

    Number_of_Assets = len(Binary_Portfolios[0])

    if Asset_Names is None:
        Asset_Names = [f"Asset_{i}" for i in range(Number_of_Assets)]

    for Binary_Portfolio in Binary_Portfolios:
        Binary_Portfolio_Vector = np.array(Binary_Portfolio, dtype=int)

        Energy_Value = compute_portfolio_energy(
            Binary_Portfolio_Vector=Binary_Portfolio_Vector,
            Q_Matrix=Q_Matrix,
        )

        Portfolio_Return = compute_portfolio_return(
            Binary_Portfolio_Vector=Binary_Portfolio_Vector,
            Expected_Return_Vector=Expected_Return_Vector,
        )

        Portfolio_Risk = compute_portfolio_risk(
            Binary_Portfolio_Vector=Binary_Portfolio_Vector,
            Sigma_Covariance_Matrix=Sigma_Covariance_Matrix,
        )

        Record = {
            "Binary_Portfolio": Binary_Portfolio,
            "Energy": Energy_Value,
            "Portfolio_Return": Portfolio_Return,
            "Portfolio_Risk": Portfolio_Risk,
        }

        for bit_index in range(Number_of_Assets):
            Record[f"{Asset_Names[bit_index]}_Included"] = Binary_Portfolio[bit_index]

        Evaluation_Records.append(Record)

    Evaluation_DataFrame = pd.DataFrame(Evaluation_Records)
    return Evaluation_DataFrame


def find_optimal_portfolio_by_energy(
    Evaluation_DataFrame: pd.DataFrame,
) -> pd.Series:
    """
    Find the portfolio with minimum energy.

    Parameters
    ----------
    Evaluation_DataFrame : pd.DataFrame

    Returns
    -------
    pd.Series
        Row corresponding to the optimal portfolio.
    """
    Optimal_Row = Evaluation_DataFrame.loc[Evaluation_DataFrame["Energy"].idxmin()]
    return Optimal_Row


def export_qubo_terms_for_qaoa(
    Q_Matrix: np.ndarray,
) -> list:
    """
    Export QUBO terms in a dictionary form suitable for QAOA or other solvers.

    Parameters
    ----------
    Q_Matrix : np.ndarray

    Returns
    -------
    dict
        Keys: 'linear_terms', 'quadratic_terms', 'Q_Matrix'.
    """
    Number_of_Assets = Q_Matrix.shape[0]

    Linear_Terms = {}
    Quadratic_Terms = {}

    for i in range(Number_of_Assets):
        Linear_Terms[i] = float(Q_Matrix[i, i])
        for j in range(i + 1, Number_of_Assets):
            if Q_Matrix[i, j] != 0.0:
                Quadratic_Terms[(i, j)] = float(Q_Matrix[i, j])

    QUBO_Terms_Dictionary = {
        "Q_Matrix": Q_Matrix,
        "linear_terms": Linear_Terms,
        "quadratic_terms": Quadratic_Terms,
    }

    return QUBO_Terms_Dictionary

