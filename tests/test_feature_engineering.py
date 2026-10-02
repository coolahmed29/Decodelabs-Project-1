"""
FILE: tests/test_feature_engineering.py
PURPOSE: Unit tests for src/feature_engineering.py
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from data_loader import load_data
from imputation import impute_categorical
from feature_engineering import (
    add_price_per_item, add_date_features, add_order_value_category,
    add_has_coupon, check_correlation_with_target
)

TEST_FILE_PATH = str(
    Path(__file__).resolve().parents[1] / "data" / "raw" / "Dataset for Data Analytics.xlsx"
)


# --- Tests for add_price_per_item() ---

def test_price_per_item_column_created():
    df = load_data(TEST_FILE_PATH)
    result = add_price_per_item(df)
    assert 'price_per_item' in result.columns


def test_price_per_item_correct_calculation():
    df = load_data(TEST_FILE_PATH)
    result = add_price_per_item(df)
    row0_expected = df['TotalPrice'].iloc[0] / df['ItemsInCart'].iloc[0]
    assert result['price_per_item'].iloc[0] == pytest.approx(row0_expected)


# --- Tests for add_date_features() ---

def test_date_features_columns_created():
    df = load_data(TEST_FILE_PATH)
    result = add_date_features(df)
    assert set(['order_month', 'order_day_of_week', 'order_quarter']).issubset(result.columns)


def test_date_features_month_in_valid_range():
    df = load_data(TEST_FILE_PATH)
    result = add_date_features(df)
    assert result['order_month'].between(1, 12).all()


# --- Tests for add_order_value_category() ---

def test_order_value_category_created():
    df = load_data(TEST_FILE_PATH)
    result = add_order_value_category(df)
    assert 'order_value_category' in result.columns


def test_order_value_category_has_three_labels():
    df = load_data(TEST_FILE_PATH)
    result = add_order_value_category(df)
    assert set(result['order_value_category'].unique()) == {'Low', 'Medium', 'High'}


# --- Tests for add_has_coupon() ---

def test_has_coupon_binary_values():
    df = load_data(TEST_FILE_PATH)
    df_imputed = impute_categorical(df.copy(), 'CouponCode', strategy='unknown', fill_value='NoCoupon')
    result = add_has_coupon(df_imputed)
    assert set(result['has_coupon'].unique()).issubset({0, 1})


# --- Tests for check_correlation_with_target() ---

def test_correlation_returns_series():
    df = load_data(TEST_FILE_PATH)
    df_fe = add_price_per_item(add_date_features(df))
    result = check_correlation_with_target(df_fe, ['price_per_item', 'order_month'], 'TotalPrice')
    assert isinstance(result, pd.Series)