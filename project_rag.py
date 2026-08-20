"""
TXL Cloud - Project knowledge retrieval (lightweight RAG)
------------------------------------
Previously, every file attached to a Project was concatenated whole into
every chat message - simple, but wasteful and noisy once a project has
more than a couple of files: irrelevant content crowds the context and
can distract the model from what the question actually needs.

This replaces that with real (if simple) retrieval: each file is split
into overlapping chunks, TF-IDF + cosine similarity ranks chunks against
the user's question, and only the most relevant ones (within a character
budget) are included. No embeddings model or GPU needed - TF-IDF is pure
math over word frequencies, fast enough to run per-message on a CPU.

Falls back to including everything when the total knowledge base is
small enough that retrieval wouldn't change anything anyway.
"""

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_CHARS = 800
CHUNK_OVERLAP = 150
SKIP_RETRIEVAL_BELOW_CHARS = 6_000  # small enough to just include everything
DEFAULT_MAX_CHARS = 6_000


def _chunk_text(text: str, filename: str) -> List[Tuple[str, str]]:
    """Splits one file's content into overlapping (source_label, chunk_text) pairs."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_CHARS:
        return [(filename, text)]
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_CHARS
        chunks.append((filename, text[start:end]))
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def retrieve(query: str, files: List[dict], max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """
    files: [{"filename": str, "content": str}, ...]
    Returns a formatted knowledge block with only the chunks most relevant
    to `query` - or everything, unchanged, if the whole knowledge base
    already fits comfortably within the budget.
    """
    total_chars = sum(len(f["content"]) for f in files)
    if total_chars <= SKIP_RETRIEVAL_BELOW_CHARS:
        return "\n\n".join(f'--- {f["filename"]} ---\n{f["content"]}' for f in files)

    all_chunks = []
    for f in files:
        all_chunks.extend(_chunk_text(f["content"], f["filename"]))
    if not all_chunks:
        return ""

    texts = [c[1] for c in all_chunks]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)
        matrix = vectorizer.fit_transform(texts + [query])
        query_vec = matrix[-1]
        chunk_vecs = matrix[:-1]
        scores = cosine_similarity(query_vec, chunk_vecs)[0]
    except ValueError:
        # e.g. query/chunks share no vocabulary after stop-word removal -
        # fall back to the first chunks of each file rather than nothing.
        scores = [0.0] * len(all_chunks)

    ranked = sorted(zip(scores, all_chunks), key=lambda x: x[0], reverse=True)

    picked = []
    used_chars = 0
    for score, (filename, chunk) in ranked:
        if score <= 0 and picked:
            break  # once relevance drops to zero, stop - don't pad with noise
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            if not picked:
                # Always return at least the single best-matching chunk,
                # truncated to fit, rather than skipping it for being too
                # big and falling back to an arbitrary, unranked one.
                picked.append((filename, chunk[:remaining]))
            break
        picked.append((filename, chunk))
        used_chars += len(chunk)

    if not picked and ranked:
        _, (filename, chunk) = ranked[0]
        picked = [(filename, chunk[:max_chars])]

    # Keep original file order for readability rather than pure relevance order.
    file_order = {f["filename"]: i for i, f in enumerate(files)}
    picked.sort(key=lambda p: file_order.get(p[0], 0))

    return "\n\n".join(f"--- {filename} (relevant excerpt) ---\n{chunk}" for filename, chunk in picked)
