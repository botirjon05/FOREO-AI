import pandas as pd
import re

def _norm(col: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(col).strip().lower()).strip("_")

def infer_schema(df: pd.DataFrame) -> dict:
    """
    Returns schema dict keyed by normalized column name.
    Each entry includes:
    -original_name
    -role: numeric | boolean | datetime | categorical | text
    -stats: min|max for numeric, sample uniques for categorical
    """

    schema = {}
    df2 = df.copy()
    df2.columns = [_norm(c) for c in df2.columns]

    for col in df2.columns:
        s = df2[col]
        entry = {
            "normalized_name": col,
            "original_name": col,
            "role": "text",
        }

        #boolean
        non_null = s.dropna()
        if len(non_null) > 0 and non_null.isin([True, False]).all():
            entry["role"] = "boolean"
            schema[col] = entry
            continue

        #numeric
        if pd.api.types.is_numeric_dtype(s):
            entry["role"] = "numeric"
            entry["min"] = float(s.min()) if len(non_null) else None
            entry["max"] = float(s.max()) if len(non_null) else None
            schema[col] = entry
            continue

        #datetime
        if pd.api.types.is_datetime64_any_dtype(s):
            entry["role"] = "datetime"
            schema[col] = entry
            continue

        #datetime: string that looks like YYYY-MM-DD
        if s.astype(str).str.match(r"^\d{4} - \d{2} - \d{2}$").mean() > 0.6:
            entry["role"] = "datetime"
            schema[col] = entry
            continue

        #categorical heuristic
        nunique = non_null.nunique()
        if nunique > 0 and nunique <= min(50, max(5, int(len(non_null) * 0.3))):
            entry["role"] = "categorical"
            entry["unique_sample"] = list(non_null.unique())[:20]
            schema[col] = entry
            continue
        schema[col] = entry

    return schema

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [_norm(c) for c in df2.columns]
    return df2


