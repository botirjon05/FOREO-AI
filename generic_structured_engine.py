import re
import pandas as pd
from dataset_schema import normalize_dataframe
from structured_intent import detect_intent

def _find_matching_rows(df: pd.DataFrame, question: str):
    """Return rows where some cell value matches a part of the question."""
    q = question.lower()
    for col in df.columns:
        # Check unique values in this column for any that appear in the question
        sample_values = df[col].astype(str).str.lower().unique()[:200]
        for val in sample_values:
            if val and val in q:
                return df[df[col].astype(str).str.lower() == val]
    return None

def _extract_date(text: str):
    match = re.search(r"\b(20\d{2}\s*-\s*\d{2}\s*-\s*\d{2})\b", text)
    if not match:
        return None
    return re.sub(r"\s*", "", match.group(1))  # -> "2025-01-01"

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9_]+", "", s)
    if s.endswith("s"):
        s = s[:-1]
    return s

def _find_col_by_mention(question: str, df: pd.DataFrame):
    q = (question or "").lower()

    # direct substring match first
    for col in df.columns:
        if col.lower() in q:
            return col

    # token-based normalized match (new_user -> new_users)
    tokens = {_norm(t) for t in re.findall(r"[a-zA-Z_]+", q)}
    for col in df.columns:
        if _norm(col) in tokens:
            return col

    # synonyms
    synonyms = {
        "revenue": ["revenue", "sales", "income"],
        "discount": ["discount", "percent", "pct"],
        "sessions": ["sessions", "visits"],
        "users": ["user", "users", "new_user", "new_users", "customers"],
        "transactions": ["transaction", "transactions", "orders", "purchases"],
    }
    for target, keywords in synonyms.items():
        if any(word in q for word in keywords):
            for col in df.columns:
                if target in col.lower():
                    return col

    return None

def _first_numeric(schema: dict):
    """Return the name of the first numeric column found in the schema, preferring common metric names."""
    preferred_keywords = ["revenue", "sales", "amount", "total", "sum", "count", "transactions"]
    numeric_cols = [col for col, meta in schema.items() if meta.get("role") == "numeric"]
    if not numeric_cols:
        return None
    # If any numeric column contains a preferred keyword, use that
    for kw in preferred_keywords:
        for col in numeric_cols:
            if kw in col.lower():
                return col
    # Otherwise, just return the first numeric column
    return numeric_cols[0]

def _first_categorical(schema: dict):
    """Return the name of the first categorical column in the schema."""
    for col, meta in schema.items():
        if meta.get("role") == "categorical":
            return col
    return None

def _first_datetime(schema: dict):
    """Return the name of the first datetime column in the schema."""
    for col, meta in schema.items():
        if meta.get("role") == "datetime":
            return col
    return None

def answer_structured(question: str, df_raw: pd.DataFrame, schema: dict):
    """
    Answer analytical questions about the DataFrame.
    Returns either an answer string or structured data in a string (with formatting).
    """
    df = normalize_dataframe(df_raw)  # Normalize the DataFrame for consistency
    intent = detect_intent(question)  # Determine the analytical intent of the question

    # 1. If the question mentions a specific value present in the DataFrame, show matching rows
    matched_rows = _find_matching_rows(df, question)
    if matched_rows is not None and not matched_rows.empty:
        return (
            "Here are the metrics for the matching row(s):\n\n"
            + "```text\n" + matched_rows.head(20).to_string(index=False) + "\n```"
        )

    # 2. If the question contains a specific date, filter data by that date
    date_val = _extract_date(question)
    date_col = _first_datetime(schema)
    if date_val and date_col and date_col in df.columns:
        filtered_by_date = df[df[date_col].astype(str) == date_val]
        if not filtered_by_date.empty:
            return (
                f"Rows for {date_val}:\n\n"
                + "```text\n" + filtered_by_date.head(50).to_string(index=False) + "\n```"
            )

    # 3. Provide schema information if asked
    if intent == "describe":
        lines = ["Detected columns:"]
        for col, meta in schema.items():
            role = meta.get("role", "unknown")
            lines.append(f"- {col} ({role})")
        return "\n".join(lines)

    # 4. Provide row count if asked
    if intent == "count":
        return f"The dataset contains {len(df)} rows."

    # 5. If question includes a date but wasn't handled above (e.g., user asks to filter by a date not present)
    if date_val:
        if date_col and date_col in df.columns:
            filtered_by_date = df[df[date_col].astype(str) == date_val]
            if filtered_by_date.empty:
                return f"No rows found for {date_val}."
            return (
                f"Rows for {date_val}:\n\n"
                + "```text\n" + filtered_by_date.head(50).to_string(index=False) + "\n```"
            )

    # 6. Determine target numeric column for aggregation questions
    target_col = _find_col_by_mention(question, df)
    numeric_col = target_col if (target_col and target_col in df.columns) else _first_numeric(schema)

    if intent in ["sum", "avg", "extreme", "compare"] and not numeric_col:
        return "I can't find any numeric columns in this dataset to compute that."

    # 7. Sum aggregation
    if intent == "sum" and numeric_col:
        total = df[numeric_col].sum()
        return f"Sum of `{numeric_col}`: {total}"

    # 8. Average aggregation
    if intent == "avg" and numeric_col:
        avg_value = df[numeric_col].mean()
        return f"Average of `{numeric_col}`: {avg_value}"

    # 9. Minimum/Maximum (extreme) value
    if intent == "extreme" and numeric_col:
        q_lower = question.lower()
        if any(keyword in q_lower for keyword in ["lowest", "minimum", "min"]):
            idx_min = df[numeric_col].idxmin()
            row_min = df.loc[idx_min]
            return (
                f"Lowest `{numeric_col}` is {row_min[numeric_col]}.\n\n"
                + "```text\n" + row_min.to_string() + "\n```"
            )
        else:
            idx_max = df[numeric_col].idxmax()
            row_max = df.loc[idx_max]
            return (
                f"Highest `{numeric_col}` is {row_max[numeric_col]}.\n\n"
                + "```text\n" + row_max.to_string() + "\n```"
            )

    # 10. Compare values between two dates or categories
    if intent == "compare" and numeric_col:
        dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", question)
        date_col = _first_datetime(schema)
        if date_col and len(dates) >= 2:
            d1, d2 = dates[0], dates[1]
            sum_d1 = df[df[date_col].astype(str) == d1][numeric_col].sum()
            sum_d2 = df[df[date_col].astype(str) == d2][numeric_col].sum()
            if sum_d1 == sum_d2:
                return f"{d1} and {d2} have equal total `{numeric_col}` (each = {sum_d1})."
            else:
                higher_date = d1 if sum_d1 > sum_d2 else d2
                return (
                    f"{higher_date} has a higher total `{numeric_col}`.\n\n"
                    + "```text\n"
                    + f"{d1}: {sum_d1}\n"
                    + f"{d2}: {sum_d2}\n"
                    + "```"
                )
        # For comparing categories (if needed, not implemented here), we could add logic similar to the above.

        return "I can compare values if you mention two specific dates or categories to compare."

    # 11. If user asks to see or filter data without specific criteria (fallback)
    if intent == "filter":
        return (
            "Showing the first 50 rows:\n\n"
            + "```text\n" + df.head(50).to_string(index=False) + "\n```"
        )

    # 12. If none of the above intents matched, prompt for a more specific question
    return "I can analyze this dataset, but I need a more specific question to answer."