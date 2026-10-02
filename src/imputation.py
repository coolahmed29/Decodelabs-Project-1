"""
FILE: src/imputation.py
PURPOSE: Handle missing values via statistical imputation (Phase 2)
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def impute_categorical(df: pd.DataFrame, column: str, strategy: str = 'unknown', fill_value: str = 'Unknown') -> pd.DataFrame:
    if strategy == 'mode':
        mode_val = df[column].mode()[0]
        df[column] = df[column].fillna(mode_val)
    elif strategy == 'unknown':
        df[column] = df[column].fillna(fill_value)
    else:
        raise ValueError("strategy must be 'mode' or 'unknown'")
    return df


def impute_numerical_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    mean_val = df[column].mean()
    df[column] = df[column].fillna(mean_val)
    return df


def impute_numerical_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    median_val = df[column].median()
    df[column] = df[column].fillna(median_val)
    return df


def impute_knn(df: pd.DataFrame, columns: list, n_neighbors: int = 5) -> pd.DataFrame:
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[columns] = imputer.fit_transform(df[columns])
    return df


def compare_before_after(df_before: pd.DataFrame, df_after: pd.DataFrame, column: str) -> dict:
    before = df_before[column]
    after = df_after[column]
    if pd.api.types.is_numeric_dtype(before.dtype):
        before_summary = {
            'mean': before.mean(),
            'median': before.median(),
            'std': before.std(),
        }
        after_summary = {
            'mean': after.mean(),
            'median': after.median(),
            'std': after.std(),
        }
    else:
        before_summary = dict(before.value_counts())
        after_summary = dict(after.value_counts())
    return {'before': before_summary, 'after': after_summary}