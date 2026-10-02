# PURPOSE: Load raw dataset and run basic validation checks

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.dtypes)
    return df


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_pct = missing_pct.round(2)
    missing_df = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
        }
    )
    return missing_df.sort_values(by="missing_pct", ascending=False)


def check_duplicates(df: pd.DataFrame, subset_col: str = "OrderID") -> dict:
    full_row_duplicates = df.duplicated().sum()
    id_duplicates = df[subset_col].duplicated().sum()
    return {
        "full_row_duplicates": full_row_duplicates,
        "id_duplicates": id_duplicates,
    }


def identify_column_types(df: pd.DataFrame) -> dict:
    numerical = []
    categorical = []
    datetime = []
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_datetime64_any_dtype(dtype):
            datetime.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            numerical.append(col)
        else:
            categorical.append(col)
    return {
        "numerical": numerical,
        "categorical": categorical,
        "datetime": datetime,
    }

"""
FILE: src/data_loader.py
PURPOSE: Load raw dataset and run basic validation/inspection checks (Phase 1)
"""

# IMPORTS NEEDED:
#   import pandas as pd


# FUNCTION 1: load_data(filepath: str) -> pd.DataFrame
#   STATUS: Done + tested
#   STEPS:
#     1. Read file using pd.read_excel(filepath)
#     2. Print/log shape (rows, columns)
#     3. Print/log dtypes (df.dtypes)
#     4. Return the dataframe
#   NOTE: Should raise FileNotFoundError naturally if path is invalid (pandas default behavior)


# FUNCTION 2: get_missing_summary(df: pd.DataFrame) -> pd.DataFrame
#   STATUS: In progress (current feature)
#   STEPS:
#     1. Compute missing count per column: df.isnull().sum()
#     2. Compute missing percentage: (missing_count / len(df)) * 100, round to 2 decimals
#     3. Combine both into one DataFrame with columns ['missing_count', 'missing_pct']
#        (index = original column names)
#     4. Sort result by 'missing_pct' descending
#     5. Return the resulting DataFrame
#   EXPECTED OUTPUT (sanity check):
#     - CouponCode: missing_count = 309, missing_pct = 25.75
#     - All other columns: missing_count = 0, missing_pct = 0.0


# FUNCTION 3: check_duplicates(df: pd.DataFrame, subset_col: str = 'OrderID') -> dict
#   STATUS: Not started yet (upcoming feature)
#   STEPS:
#     1. Count full-row duplicates: df.duplicated().sum()
#     2. Count duplicate values in the unique-id column (subset_col): df[subset_col].duplicated().sum()
#     3. Return a dict: {'full_row_duplicates': ..., 'id_duplicates': ...}
#   EXPECTED OUTPUT (sanity check):
#     - full_row_duplicates = 0, id_duplicates = 0


# FUNCTION 4: identify_column_types(df: pd.DataFrame) -> dict
#   STATUS: Not started yet (upcoming feature)
#   STEPS:
#     1. Separate columns into 'numerical' (int64/float64), 'categorical' (object/str),
#        and 'datetime' (datetime64) using df.dtypes
#     2. Return a dict: {'numerical': [...], 'categorical': [...], 'datetime': [...]}
#   EXPECTED OUTPUT (sanity check):
#     - numerical: ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice']
#     - datetime: ['Date']
#     - categorical: remaining columns (OrderID, CustomerID, Product, ShippingAddress,
#       PaymentMethod, OrderStatus, TrackingNumber, CouponCode, ReferralSource)