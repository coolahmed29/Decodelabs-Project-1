"""
FILE: tests/test_outlier_handler.py
PURPOSE: Unit tests for src/outlier_handler.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from data_loader import load_data
from outlier_handler import (
    detect_outliers_zscore, detect_outliers_iqr, get_outlier_bounds_iqr,
    cap_outliers_iqr, remove_outliers_iqr, outlier_summary
)

TEST_FILE_PATH = str(
    Path(__file__).resolve().parents[1] / "data" / "raw" / "Dataset for Data Analytics.xlsx"
)


# --- Tests for detect_outliers_zscore() ---

def test_zscore_detects_known_outlier():
    dummy_df = pd.DataFrame({'x': [10, 12, 11, 13, 10, 500]})
    result = detect_outliers_zscore(dummy_df, 'x', threshold=2)
    assert 500 in result['x'].values


def test_zscore_no_outliers_when_uniform():
    dummy_df = pd.DataFrame({'x': [10, 11, 10, 11, 10, 11]})
    result = detect_outliers_zscore(dummy_df, 'x')
    assert len(result) == 0


# --- Tests for detect_outliers_iqr() ---

def test_iqr_detects_known_outlier():
    dummy_df = pd.DataFrame({'x': [10, 12, 11, 13, 10, 500]})
    result = detect_outliers_iqr(dummy_df, 'x')
    assert 500 in result['x'].values


def test_iqr_no_outliers_when_uniform():
    dummy_df = pd.DataFrame({'x': [10, 11, 10, 11, 10, 11]})
    result = detect_outliers_iqr(dummy_df, 'x')
    assert len(result) == 0


# --- Tests for get_outlier_bounds_iqr() ---

def test_get_outlier_bounds_returns_tuple():
    df = load_data(TEST_FILE_PATH)
    bounds = get_outlier_bounds_iqr(df, 'UnitPrice')
    assert isinstance(bounds, tuple) and len(bounds) == 2
    assert bounds[0] < bounds[1]


# --- Tests for cap_outliers_iqr() ---

def test_cap_outliers_clips_values_within_bounds():
    dummy_df = pd.DataFrame({'x': [10, 12, 11, 13, 10, 500]})
    result = cap_outliers_iqr(dummy_df, 'x')
    lower, upper = get_outlier_bounds_iqr(dummy_df, 'x')
    assert result['x'].max() <= upper
    assert result['x'].min() >= lower


def test_cap_outliers_preserves_row_count():
    dummy_df = pd.DataFrame({'x': [10, 12, 11, 13, 10, 500]})
    result = cap_outliers_iqr(dummy_df, 'x')
    assert len(result) == len(dummy_df)


# --- Tests for remove_outliers_iqr() ---

def test_remove_outliers_drops_outlier_rows():
    dummy_df = pd.DataFrame({'x': [10, 12, 11, 13, 10, 500]})
    result = remove_outliers_iqr(dummy_df, 'x')
    assert 500 not in result['x'].values
    assert len(result) < len(dummy_df)


# --- Tests for outlier_summary() ---

def test_outlier_summary_returns_dataframe():
    df = load_data(TEST_FILE_PATH)
    result = outlier_summary(df, ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice'])
    assert isinstance(result, pd.DataFrame)


def test_outlier_summary_has_expected_columns():
    df = load_data(TEST_FILE_PATH)
    result = outlier_summary(df, ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice'])
    assert list(result.columns) == ['outlier_count', 'outlier_pct']


def test_outlier_summary_has_all_requested_columns_as_index():
    df = load_data(TEST_FILE_PATH)
    result = outlier_summary(df, ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice'])
    assert set(result.index) == {'Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice'}