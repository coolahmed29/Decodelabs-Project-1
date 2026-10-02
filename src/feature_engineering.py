"""
FILE: src/feature_engineering.py
PURPOSE: Create new predictive features (Phase 4)
"""

import numpy as np
import pandas as pd


def add_price_per_item(df: pd.DataFrame) -> pd.DataFrame:
    df['price_per_item'] = df['TotalPrice'] / df['ItemsInCart'].replace(0, np.nan)
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    df['order_month'] = df['Date'].dt.month
    df['order_day_of_week'] = df['Date'].dt.day_name()
    df['order_quarter'] = df['Date'].dt.quarter
    return df


def add_order_value_category(df: pd.DataFrame) -> pd.DataFrame:
    df['order_value_category'] = pd.qcut(df['TotalPrice'], q=3, labels=['Low', 'Medium', 'High'])
    return df


def add_has_coupon(df: pd.DataFrame) -> pd.DataFrame:
    df['has_coupon'] = df['CouponCode'].apply(lambda x: 0 if x == 'NoCoupon' else 1)
    return df


def check_correlation_with_target(df: pd.DataFrame, feature_cols: list, target_col: str = 'TotalPrice') -> pd.Series:
    numeric_features = [f for f in feature_cols if pd.api.types.is_numeric_dtype(df[f])]
    correlations = df[numeric_features + [target_col]].corr()[target_col].drop(target_col)
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)