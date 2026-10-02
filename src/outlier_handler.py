"""
FILE: src/outlier_handler.py
PURPOSE: Detect and treat outliers in numerical columns (Phase 3)
"""

import numpy as np
import pandas as pd


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3) -> pd.DataFrame:
    z_scores = abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores > threshold]


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    lower_bound, upper_bound = get_outlier_bounds_iqr(df, column)
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]


def get_outlier_bounds_iqr(df: pd.DataFrame, column: str) -> tuple:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (lower_bound, upper_bound)


def cap_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    lower_bound, upper_bound = get_outlier_bounds_iqr(df, column)
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    lower_bound, upper_bound = get_outlier_bounds_iqr(df, column)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def outlier_summary(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    rows = []
    for col in columns:
        count = len(detect_outliers_iqr(df, col))
        pct = round((count / len(df)) * 100, 2)
        rows.append((col, count, pct))
    return pd.DataFrame(rows, columns=['column', 'outlier_count', 'outlier_pct']).set_index('column')