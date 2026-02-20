import re

def detect_intent(question: str) -> str:
    q = (question or "").lower()

    def has_word(*words: str) -> bool:
        return any(re.search(rf"\b{re.escape(w)}\b", q) for w in words)

    if has_word("top", "most", "best"):
        return "top"
    if has_word("bottom", "least", "worst"):
        return "bottom"

    if has_word("highest", "lowest", "maximum", "minimum", "max", "min"):
        return "extreme"

    if has_word("average", "mean"):
        return "avg"

    if has_word("sum", "total"):
        return "sum"

    if has_word ("count") or "how many" in q:
        return "count"

    if has_word("comapre", "vs", "versus", "difference", "between"):
        return "compare"

    if has_word("show", "list", "rows", "filter", "where"):
        return "filter"

    return "describe"
