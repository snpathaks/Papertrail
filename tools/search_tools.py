import requests
import arxiv
import config

def search_arxiv(query: str, max_results: int = 8) -> list[dict]:
    import concurrent.futures
    def _do():
        client = arxiv.Client(page_size=min(max_results, 15), delay_seconds=0.1, num_retries=0)
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        f = ex.submit(_do)
        try:
            return f.result(timeout=3.0)
        except Exception:
            return []

def search_openalex(query: str, max_results: int = 8) -> list[dict]:
    headers = {"User-Agent": "PaperTrail-Researcher/2.0 (mailto:papertrail.academic@gmail.com)"}
    try:
        r = requests.get("https://api.openalex.org/works", timeout=8, headers=headers, params={
            "search": query, "per-page": max_results,
            "filter": "has_abstract:true"})
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
            "authors": [a["author"]["display_name"] for a in w.get("authorships", []) if a.get("author") and a["author"].get("display_name")],
            "year": w.get("publication_year"), "abstract": abstract,
            "pdf_url": (w.get("best_oa_location") or {}).get("pdf_url"),
            "citations": w.get("cited_by_count"), "source": "openalex"})
    return out

def search_crossref(query: str, max_results: int = 8) -> list[dict]:
    import re
    headers = {"User-Agent": f"PaperTrail/1.0 (mailto:{config.CONTACT_EMAIL})"}
    try:
        r = requests.get("https://api.crossref.org/works",
                         params={"query": query, "rows": max_results},
                         headers=headers, timeout=12)
        if r.status_code != 200:
            return []
        items = r.json().get("message", {}).get("items", [])
    except Exception:
        return []
    out = []
    for it in items:
        titles = it.get("title", [])
        if not titles:
            continue
        title = titles[0]
        authors = []
        for a in it.get("author", []):
            given = a.get("given", "")
            family = a.get("family", "")
            name = f"{given} {family}".strip()
            if name:
                authors.append(name)
        issued = it.get("issued", {}).get("date-parts", [[None]])[0]
        year = issued[0] if issued else None
        abstract = it.get("abstract", "")
        if abstract:
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()
        out.append({
            "id": it.get("DOI", title),
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "pdf_url": (it.get("link", [{}])[0].get("URL") if it.get("link") else None),
            "citations": it.get("is-referenced-by-count", 0),
            "source": "crossref"
        })
    return out
