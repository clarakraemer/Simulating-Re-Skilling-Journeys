import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau, pearsonr, spearmanr


def rmse(predictions, targets):
    return np.sqrt((np.nanmean(predictions - targets) ** 2))


# correct if the population S.D. is expected to be equal for the two groups.
def cohen_d(x, y):
    """
    Correct if the population S.D. is expected to be equal for the two groups.

    Source: https://stackoverflow.com/questions/21532471/how-to-calculate-cohens-d-in-python
    Parameters
    ----------
    x
    y

    Returns
    -------

    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.nanmean(x) - np.nanmean(y)) / np.sqrt(
        ((nx - 1) * np.nanstd(x, ddof=1) ** 2 + (ny - 1) * np.nanstd(y, ddof=1) ** 2)
        / dof
    )


def kendall_pval(x, y):
    return kendalltau(x, y)[1]


def pearsonr_pval(x, y):
    return pearsonr(x, y)[1]


def spearmanr_pval(x, y):
    return spearmanr(x, y)[1]


def correlation_matrix(df):
    """
    Given a pd.DataFrame, calculate Pearson's R
    and the p-values of the correlation.
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing numeric values.
    Returns
    -------
    (correlations, p_values) : tuple
        Tuple of dataframes.
    """
    df = df.dropna(axis=0)._get_numeric_data()
    cols = pd.DataFrame(columns=df.columns)

    correlations = cols.transpose().join(cols, how="outer")
    p_values = cols.transpose().join(cols, how="outer")

    for r in df.columns:
        for c in df.columns:
            # pearsonr returns a tuple like (corr, pval)
            correlations[r][c] = round(pearsonr(df[r], df[c])[0], 4)
            p_values[r][c] = round(pearsonr(df[r], df[c])[1], 4)

    return (correlations.apply(pd.to_numeric), p_values.apply(pd.to_numeric))


def naniqr(x, p_low=25, p_high=75):
    q_high, q_low = np.nanpercentile(x, q=[p_high, p_low])
    return q_high - q_low


def pbsr(df, col_x, col_y):
    """
    Compute PBS R.

    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pointbiserialr.html

    Parameters
    ----------
    df
    col_x
    col_y

    Returns
    -------

    """
    x = df[col_x]

    results_dict = {}
    for col in col_y:
        y = df[col]
        r, p = stats.pointbiserialr(x=x, y=y)

        # new_colname = "{}_{}".format(col, col_x)
        results_dict[col] = [r, p]

    return pd.DataFrame.from_dict(
        results_dict, orient="index", columns=["pbsr", "pval"]
    )
