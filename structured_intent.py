import re

def detect_intent(question: str) -> str:
    q = (question or "").lower().strip()

    # ---- HIGH PRIORITY: compare ----
    if any(k in q for k in ["compare", "vs", "versus", "difference", "delta"]):
        return "compare"

    # ---- group ranking: top/bottom by ----
    if ("by" in q) and any(k in q for k in ["top", "most", "best", "highest", "max"]):
        return "group_top"
    if ("by" in q) and any(k in q for k in ["bottom", "least", "worst", "lowest", "min"]):
        return "group_bottom"

    # ---- aggregations ----
    if any(k in q for k in ["sum", "total"]):
        return "sum"
    if any(k in q for k in ["average", "avg", "mean"]):
        return "avg"
    if any(k in q for k in ["highest", "lowest", "maximum", "minimum", "max", "min"]):
        return "extreme"
    if any(k in q for k in ["count", "how many"]):
        return "count"

    # ---- filter / show rows ----
    if any(k in q for k in ["show", "list", "rows", "filter", "where"]):
        return "filter"

    # ---- describe ONLY if explicitly asked ----
    if any(k in q for k in ["columns", "variables", "fields", "schema"]):
        return "describe"

    return "unknown"