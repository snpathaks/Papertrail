import re, requests, pymupdf4llm
import config

HEADING = re.compile(r"^#{1,4}\s*(?:\*\*)?(?:\d+(?:\.\d+)*\.?\s*)?([A-Za-z][A-Za-z \-&]{2,60}?)(?:\*\*)?\s*$")

def download_pdf(url, uid):
    path = config.PDF_DIR / f"{uid}.pdf"
    if path.exists():
        return path
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "PaperTrail/1.0"})
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            return None
        path.write_bytes(r.content)
        return path
    except requests.RequestException:
        return None

def parse_pdf(path, max_chars=1500):
    """Return chunks: [{section, page, text}] keeping section names and page numbers."""
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    chunks, section, buf, buf_page = [], "front matter", "", 1

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append({"section": section, "page": buf_page, "text": buf.strip()})
        buf = ""

    for pno, pg in enumerate(pages, start=1):
        for line in pg["text"].splitlines():
            m = HEADING.match(line.strip())
            if m:
                flush()
                section = m.group(1).strip().lower()
                buf_page = pno
                continue
            if not buf:
                buf_page = pno
            buf += line + "\n"
            if len(buf) >= max_chars:
                flush()
    flush()
    return chunks

def get_chunks(paper, uid):
    """PDF chunks if possible, otherwise fall back to the abstract."""
    if paper.get("pdf_url"):
        path = download_pdf(paper["pdf_url"], uid)
        if path:
            try:
                chunks = parse_pdf(path)
                if chunks:
                    return chunks
            except Exception:
                pass
    return [{"section": "abstract", "page": 1, "text": paper["abstract"]}]