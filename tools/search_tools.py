import requests
import arxiv
import config

def search_arxiv(query: str, max_results: int = 8) -> list[dict]:
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        out = []
        for r in client.results(search):
            out.append({
                "id": r.entry_id,
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "year": r.published.year if r.published else None,
                "abstract": r.summary.replace("\n", " "),
                "pdf_url": r.pdf_url,
                "citations": 0,
                "source": "arxiv",
            })
        return out
    except Exception:
        return []

def search_openalex(query: str, max_results: int = 8) -> list[dict]:
    try:
        r = requests.get("https://api.openalex.org/works", timeout=20, params={
            "search": query, "per-page": max_results,
            "filter": "has_abstract:true", "mailto": config.CONTACT_EMAIL})
        r.raise_for_status()
    except requests.RequestException:
        return []
    out = []
    for w in r.json().get("results", []):
        inv = w.get("abstract_inverted_index")
        if not inv:
            continue
        abstract = " ".join(word for _, word in sorted(
            (pos, word) for word, poss in inv.items() for pos in poss))
        out.append({
            "id": w["id"], "title": w["title"],
            "authors": [a["author"]["display_name"] for a in w.get("authorships", [])],
            "year": w.get("publication_year"), "abstract": abstract,
            "pdf_url": (w.get("best_oa_location") or {}).get("pdf_url"),
            "citations": w.get("cited_by_count"), "source": "openalex"})
    return out