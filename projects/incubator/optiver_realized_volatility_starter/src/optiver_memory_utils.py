from __future__ import annotations

import numpy as np
import pandas as pd


def downcast_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float columns to float32 in-place and return the same frame."""
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    if len(float_cols) == 0:
        return df
    df.loc[:, float_cols] = df.loc[:, float_cols].astype(np.float32)
    return df


def downcast_int_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast integer columns to smaller signed integer dtypes when possible."""
    int_cols = df.select_dtypes(include=["int64", "int32"]).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def optimize_tabular_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light-weight dtype optimization suitable for Optiver tabular features."""
    downcast_float_columns(df)
    downcast_int_columns(df)
    return df
