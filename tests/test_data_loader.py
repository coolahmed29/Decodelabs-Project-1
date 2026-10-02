"""
FILE: tests/test_data_loader.py
PURPOSE: Unit tests for src/data_loader.py
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from data_loader import load_data, get_missing_summary, check_duplicates, identify_column_types

TEST_FILE_PATH = str(
    Path(__file__).resolve().parents[1] / "data" / "raw" / "Dataset for Data Analytics.xlsx"
)


# --- Tests for load_data() ---

def test_load_data_returns_dataframe():
    df = load_data(TEST_FILE_PATH)
    assert isinstance(df, pd.DataFrame)


def test_load_data_correct_shape():
    df = load_data(TEST_FILE_PATH)
    assert df.shape == (1200, 14)


def test_load_data_has_expected_columns():
    df = load_data(TEST_FILE_PATH)
    expected_cols = ['OrderID', 'Date', 'CustomerID', 'Product', 'Quantity',
                     'UnitPrice', 'ShippingAddress', 'PaymentMethod', 'OrderStatus',
                     'TrackingNumber', 'ItemsInCart', 'CouponCode', 'ReferralSource', 'TotalPrice']
    assert list(df.columns) == expected_cols


def test_load_data_invalid_path_raises_error():
    with pytest.raises(FileNotFoundError):
        load_data('nonexistent_file.xlsx')


# --- Tests for get_missing_summary() ---

def test_missing_summary_returns_dataframe():
    df = load_data(TEST_FILE_PATH)
    result = get_missing_summary(df)
    assert isinstance(result, pd.DataFrame)


def test_missing_summary_correct_columns():
    df = load_data(TEST_FILE_PATH)
    result = get_missing_summary(df)
    assert list(result.columns) == ['missing_count', 'missing_pct']


def test_missing_summary_known_values():
    df = load_data(TEST_FILE_PATH)
    result = get_missing_summary(df)
    assert result.loc['CouponCode', 'missing_count'] == 309
    assert result.loc['CouponCode', 'missing_pct'] == 25.75


def test_missing_summary_other_columns_are_zero():
    df = load_data(TEST_FILE_PATH)
    result = get_missing_summary(df)
    other_cols = [col for col in df.columns if col != 'CouponCode']
    for col in other_cols:
        assert result.loc[col, 'missing_count'] == 0


# --- Tests for check_duplicates() ---

def test_check_duplicates_returns_dict():
    df = load_data(TEST_FILE_PATH)
    result = check_duplicates(df, subset_col='OrderID')
    assert isinstance(result, dict)


def test_check_duplicates_known_values():
    df = load_data(TEST_FILE_PATH)
    result = check_duplicates(df, subset_col='OrderID')
    assert result['full_row_duplicates'] == 0
    assert result['id_duplicates'] == 0


# --- Tests for identify_column_types() ---

def test_identify_column_types_returns_dict():
    df = load_data(TEST_FILE_PATH)
    result = identify_column_types(df)
    assert isinstance(result, dict)
    assert 'numerical' in result
    assert 'categorical' in result
    assert 'datetime' in result


def test_identify_column_types_known_values():
    df = load_data(TEST_FILE_PATH)
    result = identify_column_types(df)
    assert 'Quantity' in result['numerical']
    assert 'Date' in result['datetime']
    assert 'Product' in result['categorical']