from typing import TypedDict, Any

class PaperTrailState(TypedDict, total=False):
    query: str
    expanded_queries: list[str]
    attempts: int
    papers: list[dict]      # search results
    selected: list[dict]    # user-approved papers
    extracted: list[dict]   # structured facts per paper
    comparison: dict
    gaps: list[dict]
    review: str
    verification: dict
    issues: list[dict]      # papers that fell back to abstract-only
    progress_callback: Any  # UI callback function(message: str)