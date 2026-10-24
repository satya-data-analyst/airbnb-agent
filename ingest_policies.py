"""
Ingest the policy documents: chunk -> embed -> cache.

    python ingest_policies.py

Reads every .txt in data/policies/, splits each into chunks, embeds them with
the same model the agent will use, and saves policy_embeddings.npz so the
agent never has to re-embed.

Why chunk at all? A whole policy is ~1,000 words. If we embed it as one blob,
the vector is an average of everything in it and matches nothing well. Small
chunks keep one idea per vector, which is what makes retrieval sharp.

Chunks carry their source file name, so the agent can cite which policy an
answer came from.
"""

import glob
import os
import sys

import numpy as np
from fastembed import TextEmbedding

POLICY_DIR = "data/policies"
OUT_PATH = "policy_embeddings.npz"
CHUNK_CHARS = 800      # roughly 130-150 words per chunk
MIN_CHUNK = 200        # don't keep tiny scraps


def load_policies():
    files = sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt")))
    if not files:
        sys.exit(
            f"No .txt files found in {POLICY_DIR}/.\n"
            "Run split_policies.py first, then try again."
        )
    docs = []
    for path in files:
        text = open(path, encoding="utf-8").read().strip()
        if not text:
            continue
        title = text.split("\n", 1)[0].strip()          # first line is the title
        docs.append((os.path.basename(path), title, text))
    return docs


def chunk_document(text, source, title):
    """Group paragraphs into ~CHUNK_CHARS chunks, keeping paragraph breaks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""

    for para in paragraphs:
        # a single oversized paragraph gets split on its own
        if len(para) > CHUNK_CHARS:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), CHUNK_CHARS):
                piece = para[i:i + CHUNK_CHARS].strip()
                if len(piece) >= MIN_CHUNK:
                    chunks.append(piece)
            continue

        if len(current) + len(para) + 2 <= CHUNK_CHARS:
            current = (current + "\n\n" + para).strip()
        else:
            if len(current) >= MIN_CHUNK:
                chunks.append(current)
            current = para

    if len(current) >= MIN_CHUNK:
        chunks.append(current)

    return [
        {"text": c, "source": source, "title": title}
        for c in chunks
    ]


def main():
    docs = load_policies()
    print(f"Loaded {len(docs)} policy documents from {POLICY_DIR}/\n")

    all_chunks = []
    for source, title, text in docs:
        chunks = chunk_document(text, source, title)
        all_chunks.extend(chunks)
        print(f"  {source:<34} {len(text):>6,} chars  ->  {len(chunks):>2} chunks")

    if not all_chunks:
        sys.exit("No chunks produced — check the policy files.")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Embedding (this is fast — the corpus is small)...")

    model = TextEmbedding()
    texts = [c["text"] for c in all_chunks]
    # embed in small batches — keeps memory low on any machine
    vectors = []
    for i in range(0, len(texts), 8):
        vectors.extend(model.embed(texts[i:i + 8], batch_size=8))
        print(f"  embedded {min(i + 8, len(texts))}/{len(texts)}", flush=True)
    vecs = np.array(vectors, dtype=float)

    np.savez(
        OUT_PATH,
        vecs=vecs,
        texts=np.array([c["text"] for c in all_chunks], dtype=object),
        sources=np.array([c["source"] for c in all_chunks], dtype=object),
        titles=np.array([c["title"] for c in all_chunks], dtype=object),
    )

    print(f"\nSaved {OUT_PATH}")
    print(f"  vectors: {vecs.shape}")
    print("\nSample chunks:")
    for c in all_chunks[:3]:
        first = c["text"].replace("\n", " ")[:90]
        print(f"  [{c['source'][:22]}] {first}...")

    print("\nDone. Next: the agent (SQL + RAG + BOTH).")


if __name__ == "__main__":
    main()
