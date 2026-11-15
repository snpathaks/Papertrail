import re

def _label(p):
    first = p["authors"][0].split()[-1] if p["authors"] else "Unknown"
    return f"{first} et al., {p['year']}"

def render_review(review, papers):
    by_ref = {p["ref"]: p for p in papers}
    text = re.sub(r"\[(P\d+)\]",
                  lambda m: f"[{_label(by_ref[m.group(1)])}]" if m.group(1) in by_ref else m.group(0),
                  review)
    refs = "\n".join(f"- {_label(p)}: {p['title']}. {p.get('id', p.get('uid', ''))}" for p in papers)
    return f"# Literature Review\n\n{text}\n\n## References\n{refs}"

def to_bibtex(papers):
    entries = []
    for p in papers:
        key = re.sub(r"\W", "", (p["authors"][0].split()[-1] if p["authors"] else "x")) + str(p.get("year", ""))
        url = p.get("id", p.get("uid", ""))
        entries.append(f"@article{{{key},\n  title = {{{p['title']}}},\n"
                       f"  author = {{{' and '.join(p.get('authors', []))}}},\n"
                       f"  year = {{{p.get('year', '')}}},\n  url = {{{url}}}\n}}")
    return "\n\n".join(entries)