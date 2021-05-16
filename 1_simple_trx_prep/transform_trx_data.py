import functools
import itertools
import logging
import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

PATH_IN = "data/1_trx_data_clean.feather"
PATH_OUT = "data/2_trx_data_prepared.parquet"


def initialize_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create console handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Create file handler
    fh = logging.FileHandler(
        "prepare_data.log", "w", encoding=None, delay="true"
    )
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def logging_runtime(func):
    """Decorator that logs time for a function call."""

    @functools.wraps(func)
    def logger_wrapper(*args, **kwargs):
        """Function that logs time."""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(
            f"Calling {func.__name__} - Elapsed time (s): {(end - start):.2f}"
        )
        return result

    return logger_wrapper


@logging_runtime
def load_and_prepare_clean_data(path: Union[Path, str]) -> pd.DataFrame:
    """Load the cleaned trx data into a dataframe, tweak some dtypes
    and - very important - sort the data by trx_type to always have
    the chronological order Activation -> Purchase -> Redemption per
    member and date. (Conveniently this is the alphabetical order.)
    """
    df = pd.read_parquet(path)
    df["member"] = df["member"].astype("object")
    df["device"] = df["device"].astype("category")
    df["trx_type"] = df["trx_type"].astype("category")
    df.sort_values(["member", "date", "trx_type"], inplace=True)

    logger.info(
        f"DataFrame has {len(df):,.0f} rows "
        f"and includes data for {df['member'].nunique():,.0f} members."
    )

    return df


@logging_runtime
def create_basic_voucher_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Create separate columns containing the values for
    voucher activations ("v_a"), voucher redemptions ("v_r")
    and all of them combined.
    """
    df["v_a"] = np.where(
        (df["device"] == "Financial Voucher") & (df["value"] > 0),
        df["value"],
        0,
    )
    df["v_r"] = np.where(
        (df["device"] == "Financial Voucher") & (df["value"] < 0),
        df["value"],
        0,
    )
    df["v"] = np.where(df["device"] == "Financial Voucher", df["value"], 0)
    return df


def _calculate_voucher_sums(v: pd.Series) -> np.ndarray:
    """Helper function to calculate the accumulated sum of
    voucher "credit" a customer has at any given time. We cannot
    kow about credit from earlier periods but we make sure that
    the sum never becomes negative.
    """
    v_sum = np.array(list(itertools.accumulate(v)))
    # Make sure that v_sum never has a negative value
    v_min = np.min(v_sum)
    if v_min < 0:
        top_up_value = v_min
        v_sum = v_sum - top_up_value
    return v_sum


@logging_runtime
def create_voucher_sum_col(df: pd.DataFrame) -> pd.DataFrame:
    """Use a groupby "window function" to insert the
    voucher sums into a new column "v_sum".
    """
    df = df.assign(
        v_sum=df.groupby(["member"])["v"].transform(_calculate_voucher_sums)
    )
    df.drop("v", axis=1, inplace=True)
    return df


@logging_runtime
def shift_and_drop_redemptions(df: pd.DataFrame) -> pd.DataFrame:
    """Shift redemtion values in "v_r" column one row up, so they
    end up in the row of the corresponing transaction. This makes
    it possible to flag the respective transactions as ones with
    redemption in a later step. Then delete all redemption rows, as
    they are no longer needed.
    """
    df = df.assign(v_r=df.groupby(["member"])["v_r"].shift(-1))
    df = df[~df["trx_type"].isin(["Redemption"])]
    df["v_r"] = df["v_r"].replace(np.nan, 0)

    # Remove "Redemption" Category
    df["trx_type"].cat.remove_unused_categories(inplace=True)
    return df


@logging_runtime
def calculate_interval_activation_to_next_purchase(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a new col "delta_a" containing the interval from each
    activation to the next purchase as int (for days). Values of
    all non-activation rows are set to NaN.
    """
    df = df.assign(delta_a=df.groupby(["member"])["date"].diff(-1) * -1)

    df["delta_a"] = np.where(
        df["trx_type"] == "Activation", df["delta_a"].dt.days, np.NaN
    )
    return df


@logging_runtime
def calculate_purchase_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Create a new col "delta_p" containing the interval from each
    purchase to the next as int (for days). Values of all non-purchase
    rows are set to NaN.
    """
    df = df.assign(
        delta_p=df.groupby(["member", "trx_type"])["date"].diff(-1) * -1
    )

    df["delta_p"] = np.where(
        df["trx_type"] == "Purchase", df["delta_p"].dt.days, np.NaN
    )
    return df


@logging_runtime
def flag_purchases_depending_on_vouchers(df: pd.DataFrame) -> pd.DataFrame:
    """Create three new boolean columns to classify purchases into
    each of the following three categories: "p_v_red" = purchase with
    redemption, "p_v_miss" = purchase without redemption (but voucher
    credit would have been available), "p_v_empty" = no voucher credit
    available.
    """
    df["p_v_red"] = np.where(
        (df["trx_type"] == "Purchase") & (df["v_r"] < 0), 1, 0
    )
    df["p_v_miss"] = np.where(
        (df["trx_type"] == "Purchase") & (df["v_r"] == 0) & (df["v_sum"] > 0),
        1,
        0,
    )
    df["p_v_empty"] = np.where(
        (df["trx_type"] == "Purchase") & (df["v_sum"] == 0), 1, 0
    )

    for col in ["p_v_red", "p_v_miss", "p_v_empty"]:
        df[col] = df[col].astype("bool")
    return df


@logging_runtime
def calculate_discount_pct(df) -> pd.DataFrame:
    """Caclulate a column "discount_pct" denoting the relative
    value of discounts. This value will be used to control for a
    threshold when setting a discount flag in the next step. (Note
    the calculation is such that discounts on returns wont reach
    the threshold.)
    """
    df["gross_value"] = df["value"] + df["discount"]
    df["discount_pct"] = df["discount"] / df["gross_value"]
    return df


@logging_runtime
def flag_purchases_depending_on_discounts(
    df: pd.DataFrame, threshold_pct: float = 0.1
) -> pd.DataFrame:
    """Create a boolean columns to classify purchases having a
    discount whose relative value to the gross transaction price
    reaches a certain threshold.
    """
    df["p_discount"] = np.where(
        df["discount_pct"] >= threshold_pct, 1, 0
    ).astype("bool")

    df.drop(["gross_value", "discount_pct"], axis=1, inplace=True)
    return df


@logging_runtime
def save_transformed_data(df, path: Union[Path, str]):
    # df = df.reset_index(drop=True)
    df.to_parquet(path)


def main(
    path_in: Union[Path, str],
    path_out: Union[Path, str],
    threshold_pct: float,
    logger,
):

    logger.info("Starting process ...")
    df = load_and_prepare_clean_data(path_in)
    df = create_basic_voucher_cols(df)
    df = create_voucher_sum_col(df)
    df = shift_and_drop_redemptions(df)
    df = calculate_interval_activation_to_next_purchase(df)
    df = calculate_purchase_interval(df)
    df = flag_purchases_depending_on_vouchers(df)
    df = calculate_discount_pct(df)
    df = flag_purchases_depending_on_discounts(df)
    save_transformed_data(df, path_out)
    logger.info("Job complete!")


if __name__ == "__main__":
    logger = initialize_logger()
    main(PATH_IN, PATH_OUT, 0.1, logger)
