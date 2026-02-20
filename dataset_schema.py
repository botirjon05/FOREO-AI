import pandas as pd
import re

from sklearn.externals.array_api_extra import nunique


def _norm(col: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(col).strip().lower()).strip("_")

def infer_schema(df: pd.DataFrame) -> dict:
    df = normalize_dataframe(df)

    schema = {}
    n = len(df)

    for c in df.columns:
        s = df[c]

        #numeric
        if pd.api.types.is_numeric_dtype(s):
            schema[c] = {"role": "numeric"}
            continue

        #datetime detection (try parse)
        if s.dtype == "object":
            dt = pd.to_datetime(s, errors="coerce", utc=False)
            if dt.notna().mean() >= 0.7:
                schema[c] = {"role": "datetime"}
                continue

        #categorical vs text
        nunique = s.nunique(dropna=True)
        if n > 0 and nunique <= max(30, int(0.2 * n)):
            schema[c] = {"role": "categorical"}
        else:
            schema[c] = {"role": "text"}
    return schema

import pandas as pd
import re

def normalize_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 1) Normalize column names
    df.columns = [
        re.sub(r"\s+", "_", str(c).strip().lower())
        for c in df.columns
    ]

    # 2) Trim strings
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip().replace({"nan": None, "None": None, "": None})

    # 3) Convert date-like columns if possible (optional)
    # (You can keep your existing date logic if you already have it)

    # 4) Strong numeric coercion (IMPORTANT FIX)
    # Convert columns that are "mostly numeric among non-null"
    for c in df.columns:
        s = df[c]

        # Skip if it's already numeric
        if pd.api.types.is_numeric_dtype(s):
            continue

        # Try to parse numeric values from strings
        # - handle comma decimals "5,0" -> "5.0"
        # - remove % if present "12%" -> "12"
        s_as_str = s.astype(str).str.strip()
        s_as_str = s_as_str.str.replace(",", ".", regex=False)
        s_as_str = s_as_str.str.replace("%", "", regex=False)

        num = pd.to_numeric(s_as_str, errors="coerce")

        non_null = s.notna().sum()
        numeric_non_null = num.notna().sum()

        # Only decide based on non-null rows
        if non_null > 0 and (numeric_non_null / non_null) >= 0.6:
            df[c] = num

    return df




