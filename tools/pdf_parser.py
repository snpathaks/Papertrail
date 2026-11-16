import re, requests, pymupdf4llm
import config

HEADING = re.compile(r"^#{1,4}\s*(?:\*\*)?(?:\d+(?:\.\d+)*\.?\s*)?([A-Za-z][A-Za-z \-&]{2,60}?)(?:\*\*)?\s*$")

def download_pdf(url, uid):
    """Download PDF with configured timeout (30s) and file size cap (15 MB).
    Returns (path, None) on success or (None, error_reason) on failure.
    """
    path = config.PDF_DIR / f"{uid}.pdf"
    if path.exists():
        if path.stat().st_size > config.PDF_MAX_SIZE_BYTES:
            return None, f"Cached file exceeded {config.PDF_MAX_SIZE_MB} MB cap ({path.stat().st_size} bytes)"
        return path, None
    try:
        with requests.get(
            url,
            timeout=config.PDF_DOWNLOAD_TIMEOUT,
            headers={"User-Agent": "PaperTrail/1.0"},
            stream=True
        ) as r:
            r.raise_for_status()
            cl = r.headers.get("Content-Length")
            if cl and int(cl) > config.PDF_MAX_SIZE_BYTES:
                return None, f"File size ({int(cl) / (1024*1024):.1f} MB) exceeds {config.PDF_MAX_SIZE_MB} MB cap"
            
            size = 0
            chunks = []
            for chunk in r.iter_content(chunk_size=65536):
                size += len(chunk)
                if size > config.PDF_MAX_SIZE_BYTES:
                    return None, f"Download exceeded {config.PDF_MAX_SIZE_MB} MB size cap"
                chunks.append(chunk)
            
            content = b"".join(chunks)
            if not content.startswith(b"%PDF"):
                return None, "File content is not a valid PDF (%PDF header missing)"
            path.write_bytes(content)
            return path, None
    except requests.exceptions.Timeout:
        return None, f"Download timed out after {config.PDF_DOWNLOAD_TIMEOUT}s"
    except requests.exceptions.RequestException as e:
        return None, f"Download failed ({type(e).__name__}): {e}"
    except Exception as e:
        return None, f"Unexpected download error: {e}"

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
    """PDF chunks if possible, otherwise fall back to the abstract.
    Returns: (chunks, issue_dict_or_None)
    """
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        return [{"section": "abstract", "page": 1, "text": paper.get("abstract", "")}], {
            "fallback": True, "reason": "No PDF URL available for paper"
        }

    path, err = download_pdf(pdf_url, uid)
    if not path:
        return [{"section": "abstract", "page": 1, "text": paper.get("abstract", "")}], {
            "fallback": True, "reason": err or "Failed to download PDF"
        }

    try:
        chunks = parse_pdf(path)
        if chunks:
            return chunks, None
        return [{"section": "abstract", "page": 1, "text": paper.get("abstract", "")}], {
            "fallback": True, "reason": "Parsed PDF returned empty text"
        }
    except Exception as e:
        return [{"section": "abstract", "page": 1, "text": paper.get("abstract", "")}], {
            "fallback": True, "reason": f"PDF parsing error: {e}"
        }