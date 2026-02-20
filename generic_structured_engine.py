import re
import pandas as pd
from dataset_schema import normalize_dataframe
from structured_intent import detect_intent


DATE_RE = re.compile(r"\b(20\d{2}\s*-\s*\d{2}\s*-\s*\d{2})\b")

def _extract_all_dates(text: str) -> list[str]:
    """Return all dates in YYYY-MM-DD (normalized, no spaces) found in text."""
    if not text:
        return []
    matches = DATE_RE.findall(text)
    return [re.sub(r"\s*", "", m) for m in matches]

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9_]+", "", s)
    if s.endswith("s"):
        s = s[:-1]
    return s

def _find_col_by_mention(question: str, df: pd.DataFrame):
    q = (question or "").lower()

    # direct substring match
    for col in df.columns:
        if col.lower() in q:
            return col

    # token-based match (new_user -> new_users)
    tokens = {_norm(t) for t in re.findall(r"[a-zA-Z_]+", q)}
    for col in df.columns:
        if _norm(col) in tokens:
            return col

    # synonyms
    synonyms = {
        "revenue": ["revenue", "sales", "income"],
        "discount": ["discount", "percent", "pct"],
        "session": ["sessions", "visits"],
        "user": ["user", "users", "new_user", "new_users", "customers"],
        "transaction": ["transaction", "transactions", "orders", "purchases"],
    }
    for target, keywords in synonyms.items():
        if any(word in q for word in keywords):
            for col in df.columns:
                if target in col.lower():
                    return col

    return None

def _first_categorical(schema: dict):
    for c, meta in schema.items():
        if meta.get("role") in ("categorical", "text"):
            return c
    return None

def _first_numeric(schema: dict):
    preferred = ["revenue", "sales", "amount", "total", "sum", "count", "transaction", "user", "session"]
    numeric_cols = [c for c, meta in schema.items() if meta.get("role") == "numeric"]
    if not numeric_cols:
        return None
    for kw in preferred:
        for c in numeric_cols:
            if kw in c.lower():
                return c
    return numeric_cols[0]

def _guess_date_col(df: pd.DataFrame, schema: dict):
    """Prefer schema datetime; fallback to column name containing 'date'."""
    # 1) schema says datetime
    for c, meta in schema.items():
        if meta.get("role") == "datetime" and c in df.columns:
            return c

    # 2) name contains 'date'
    for c in df.columns:
        if "date" in c.lower():
            return c

    return None

def _find_matching_rows(df: pd.DataFrame, question: str):
    """Row match for exploration: match any categorical cell value mentioned in question."""
    q = (question or "").lower()
    for col in df.columns:
        sample_values = df[col].astype(str).str.lower().unique()[:200]
        for val in sample_values:
            # skip extremely short tokens (prevents weird matches like '01', '1', etc.)
            if not val or len(val) < 3:
                continue
            if val in q:
                return df[df[col].astype(str).str.lower() == val]
    return None

TOP_BOTTOM_RE = re.compile(r"\b(top|bottom)\s+([a-zA-Z_]+)\s+by\s+([a-zA-Z_]+)\b")
ON_RE = re.compile(r"\bon\s+([a-zA-Z0-9_ -]+)\b")  # "on tiktok", "on TV", etc.

def _resolve_col(name: str, df, schema, allowed_roles = None):
    """Resolve a word like "creator" to an actual column name"""

    if not name:
        return None
    n = name.lower().strip()

    #exact match
    for c in df.columns:
        if c.lower() == n:
            if allowed_roles is None or schema.get(c, {}).get("role") in allowed_roles:
                return c

    #fuzzy contains
    for c in df.columns:
        if n in c.lower():
            if allowed_roles is None or schema.get(c, {}).get("role") in allowed_roles:
                return c
    return None

def _apply_on_filter(question: str, df, schema):

    q = (question or "").lower()
    m = ON_RE.search(q)
    if not m:
        return df, None, None

    preffered = []
    for key in ["platform", "source", "team", "owner", "creator", "product", "placement"]:
        col = _resolve_col(key, df, schema, allowed_roles = {"categorical", "text"})
        if col:
            preffered.append(col)

    candidates = preffered + [
        c for c in df.columns
        if c not in preffered and schema.get(c, {}).get("role") in ("categorical", "text")
    ]

    for col in candidates:
        ser = df[col].astype(str).str.strip().str.lower()
        mask = ser == raw_val.lower()
        if mask.any():
            return df[mask], col, raw_val

    return df.iloc[0:0], None, raw_val


def answer_structured(question: str, df_raw: pd.DataFrame, schema: dict):
    df = normalize_dataframe(df_raw)
    intent = detect_intent(question)

    # ---------- ANALYSIS INTENTS FIRST ----------
    # If user is doing analysis (compare/sum/avg/extreme/count),
    # do NOT hijack with row-matching or "show rows for date".
    analysis_intents = {"compare", "sum", "avg", "extreme", "count"}

    # Always allow describe
    if intent == "describe":
        lines = ["Detected columns:"]
        for col, meta in schema.items():
            lines.append(f"- {col} ({meta.get('role','unknown')})")
        return "\n".join(lines)

    if intent == "count":
        return f"The dataset contains {len(df)} rows."

    # Find metric column for numeric analysis
    metric_col = _find_col_by_mention(question, df)
    if not metric_col:
        metric_col = _first_numeric(schema)

    if intent in {"sum", "avg", "extreme", "compare"} and not metric_col:
        return "I can't find any numeric column to compute that. Try: “what columns do I have?”"

    # Compare between two dates (sum per date)
    if intent == "compare":
        dates = _extract_all_dates(question)
        date_col = _guess_date_col(df, schema)

        if not date_col:
            return "I can compare two dates, but I couldn't identify a date column in this dataset."

        if len(dates) < 2:
            return "Please include two dates like: compare new_users between 2025-01-01 and 2025-01-05"

        d1, d2 = dates[0], dates[1]

        s1 = df[df[date_col].astype(str) == d1][metric_col].sum()
        s2 = df[df[date_col].astype(str) == d2][metric_col].sum()

        if s1 == 0 and s2 == 0:
            # helpful hint: maybe dates not present
            return f"I couldn't find rows for {d1} or {d2} (date column: {date_col})."

        if s1 == s2:
            return f"{d1} and {d2} are equal for `{metric_col}` (sum={s1})."

        better = d1 if s1 > s2 else d2
        return (
            f"{better} is higher for `{metric_col}` (sum).\n\n"
            f"```text\n{d1}: {s1}\n{d2}: {s2}\n```"
        )

    # Sum
    if intent == "sum":
        total = df[metric_col].sum()
        return f"Sum of `{metric_col}`: {total}"

    # Average
    if intent == "avg":
        val = df[metric_col].mean()
        return f"Average of `{metric_col}`: {val}"

    # Extreme
    if intent == "extreme":
        q = question.lower()

        df2, on_col, on_val = _apply_on_filter(question, df, schema)
        if df2.empty and on_val:
            plat = _resolve_col("platform", df, schema, allowed_roles={"categorical", "text"})
            if plat:
                vals = sorted(set(df[plat].astype(str).str.strip().unique()))[:20]
                return f"No rows match `{on_val}` in `{plat}`. Available {plat} values (first 20): {vals}"
            return f"No rows match `{on_val}` in any categorical column."

        if any(k in q for k in ["lowest", "minimum", "min"]):
            idx = df2[metric_col].idxmin()
            row = df2.loc[idx]
            return f"Lowest `{metric_col}` is {row[metric_col]}.\n\n```text\n{row.to_string()}\n```"
        else:
            idx = df2[metric_col].idxmax()
            row = df2.loc[idx]
            return f"Highest `{metric_col}` is {row[metric_col]}.\n\n```text\n{row.to_string()}\n```"

    # ---- TOP / BOTTOM ----
    if intent in ("top", "bottom"):
        # metric
        metric_col = _find_col_by_mention(question, df) or _first_numeric(schema)
        if not metric_col:
            return "I can't find a numeric metric to rank by."

        # parse: "top <group> by <metric>"
        m = TOP_BOTTOM_RE.search((question or "").lower())
        group_word = m.group(2) if m else None
        metric_word = m.group(3) if m else None

        # if metric word exists, prefer resolving it
        if metric_word:
            resolved_metric = _resolve_col(metric_word, df, schema, allowed_roles={"numeric"})
            if resolved_metric:
                metric_col = resolved_metric

        # group column
        group_col = None
        if group_word:
            group_col = _resolve_col(group_word, df, schema, allowed_roles={"categorical", "text"})

        # fallback if not found
        if not group_col:
            # pick a categorical column, but avoid url-like columns if possible
            cat_cols = [c for c, meta in schema.items() if meta.get("role") in ("categorical", "text")]
            non_url = [c for c in cat_cols if "url" not in c.lower() and "link" not in c.lower()]
            group_col = (non_url[0] if non_url else (cat_cols[0] if cat_cols else None))

        if not group_col:
            return "I can't find a categorical column to rank."

        # apply optional "on <value>" filter
        df2, on_col, on_val = _apply_on_filter(question, df, schema)
        if df2.empty and on_val:
            # helpful message: show available unique values for platform/source if we can
            plat = _resolve_col("platform", df, schema, allowed_roles={"categorical", "text"})
            if plat:
                vals = sorted(set(df[plat].astype(str).str.strip().unique()))[:20]
                return f"No rows match `{on_val}` in `{plat}`. Available {plat} values (first 20): {vals}"
            return f"No rows match `{on_val}` in any categorical column."

        agg = (
            df2.groupby(group_col, dropna=False)[metric_col]
            .sum()
            .sort_values(ascending=(intent == "bottom"))
        )

        if agg.empty:
            return "No data available to compute top/bottom."

        best_key = agg.index[0]
        best_val = agg.iloc[0]
        direction = "Top" if intent == "top" else "Bottom"
        preview_txt = agg.head(10).to_string()

        scope = f" (filtered on {on_col}={on_val})" if on_col else ""
        return (
            f"{direction} '{group_col}' by '{metric_col}' {scope}: **{best_key}** = {best_val}\n\n"
            f"```text\n{preview_txt}\n```"
        )

    # ---------- EXPLORATION MODES AFTER ----------
    # filter/show/list: allow row matching and date filtering
    if intent == "filter":
        matched = _find_matching_rows(df, question)
        if matched is not None and not matched.empty:
            return "Here are the matching row(s):\n\n```text\n" + matched.head(50).to_string(index=False) + "\n```"

        dates = _extract_all_dates(question)
        if dates:
            date_col = _guess_date_col(df, schema)
            if date_col:
                sub = df[df[date_col].astype(str) == dates[0]]
                if not sub.empty:
                    return f"Rows for {dates[0]}:\n\n```text\n{sub.head(50).to_string(index=False)}\n```"
                return f"No rows found for {dates[0]}."

        return "Showing the first 50 rows:\n\n```text\n" + df.head(50).to_string(index=False) + "\n```"

    # default fallback
    return "I can analyze this dataset. Try: compare two dates, highest/lowest, average, sum, or count rows."