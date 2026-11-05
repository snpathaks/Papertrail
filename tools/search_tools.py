import config

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