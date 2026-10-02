"""
FILE: tests/test_imputation.py
PURPOSE: Unit tests for src/imputation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from data_loader import load_data
from imputation import (
    impute_categorical, impute_numerical_mean, impute_numerical_median,
    impute_knn, compare_before_after
)

TEST_FILE_PATH = str(
    Path(__file__).resolve().parents[1] / "data" / "raw" / "Dataset for Data Analytics.xlsx"
)


# --- Tests for impute_categorical() ---

def test_impute_categorical_fills_all_nan():
    df = load_data(TEST_FILE_PATH)
    result = impute_categorical(df, 'CouponCode', strategy='unknown', fill_value='NoCoupon')
    assert result['CouponCode'].isnull().sum() == 0


def test_impute_categorical_unknown_strategy_correct_count():
    df = load_data(TEST_FILE_PATH)
    result = impute_categorical(df, 'CouponCode', strategy='unknown', fill_value='NoCoupon')
    assert result['CouponCode'].value_counts()['NoCoupon'] == 309


def test_impute_categorical_mode_strategy():
    df = load_data(TEST_FILE_PATH)
    result = impute_categorical(df.copy(), 'CouponCode', strategy='mode')
    assert result['CouponCode'].isnull().sum() == 0


def test_impute_categorical_invalid_strategy_raises_error():
    df = load_data(TEST_FILE_PATH)
    with pytest.raises(ValueError):
        impute_categorical(df, 'CouponCode', strategy='bogus')


# --- Tests for impute_numerical_mean() ---

def test_impute_numerical_mean_fills_nan():
    dummy_df = pd.DataFrame({'x': [1, 2, np.nan, 4]})
    result = impute_numerical_mean(dummy_df, 'x')
    assert result['x'].isnull().sum() == 0
    assert result['x'].iloc[2] == pytest.approx((1 + 2 + 4) / 3)


# --- Tests for impute_numerical_median() ---

def test_impute_numerical_median_fills_nan():
    dummy_df = pd.DataFrame({'x': [1, 2, np.nan, 100]})
    result = impute_numerical_median(dummy_df, 'x')
    assert result['x'].isnull().sum() == 0
    assert result['x'].iloc[2] == dummy_df['x'].median()


# --- Tests for impute_knn() ---

def test_impute_knn_fills_nan():
    dummy_df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [10, 20, 30, 40]})
    result = impute_knn(dummy_df, ['a', 'b'], n_neighbors=2)
    assert result['a'].isnull().sum() == 0


# --- Tests for compare_before_after() ---

def test_compare_before_after_returns_dict():
    df_before = load_data(TEST_FILE_PATH)
    df_after = impute_categorical(df_before.copy(), 'CouponCode', strategy='unknown', fill_value='NoCoupon')
    result = compare_before_after(df_before, df_after, 'CouponCode')
    assert 'before' in result and 'after' in result


def test_compare_before_after_shows_nocoupon_in_after_only():
    df_before = load_data(TEST_FILE_PATH)
    df_after = impute_categorical(df_before.copy(), 'CouponCode', strategy='unknown', fill_value='NoCoupon')
    result = compare_before_after(df_before, df_after, 'CouponCode')
    assert 'NoCoupon' not in result['before']
    assert 'NoCoupon' in result['after']