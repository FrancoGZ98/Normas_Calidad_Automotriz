"""
Diagnóstico de recuperación (retrieval) — NO llama al LLM ni gasta cuota.

Para cada pregunta del dataset que cita páginas en "source", revisa si alguna
de esas páginas aparece entre los k fragmentos recuperados (hit@k).
Sirve para separar dos problemas distintos:
  - hit@k bajo  → el retriever no encuentra el fragmento correcto
                  (embeddings, chunking o k).
  - hit@k alto pero correctness bajo → el problema está en el prompt o en el LLM.

Uso:
    python -m app.diagnostico_retrieval
    EMBEDDINGS_MODEL=intfloat/multilingual-e5-large CHUNK_SIZE=1000 python -m app.diagnostico_retrieval

Variables (opcionales): EMBEDDINGS_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, DIAG_KS (ej. "4,8")
"""

import json
import os
import re
import time

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag_pipeline import load_documents

DATASET_PATH = "tests/eval_dataset.json"
MODEL = os.getenv("EMBEDDINGS_MODEL") or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 512)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 50)
KS = [int(k) for k in (os.getenv("DIAG_KS") or "4,8").split(",")]


def expected_pages(source):
    """'a.pdf, pp. 11-12; b.pdf, p. 7' → {('a.pdf', 11), ('a.pdf', 12), ('b.pdf', 7)}"""
    pages = set()
    for fname, spec in re.findall(r"([\w\-.]+\.pdf),\s*pp?\.\s*([\d,\s\-]+)", source):
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-")
                pages.update((fname, p) for p in range(int(a), int(b) + 1))
            elif part:
                pages.add((fname, int(part)))
    return pages


def doc_page(doc):
    """Metadatos de PyPDFLoader (página base 0) → ('archivo.pdf', página base 1)."""
    return os.path.basename(doc.metadata["source"]), int(doc.metadata["page"]) + 1


def main():
    print("=" * 78)
    print(f"🔎 Modelo de embeddings: {MODEL}")
    print(f"✂️  chunk_size={CHUNK_SIZE} | chunk_overlap={CHUNK_OVERLAP} | k evaluados={KS}")

    t0 = time.time()
    docs = load_documents()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    print(f"📄 Páginas: {len(docs)} | Fragmentos: {len(chunks)}")

    vectordb = FAISS.from_documents(chunks, embedding=FastEmbedEmbeddings(model_name=MODEL))
    print(f"⏱️  Índice construido en {time.time() - t0:.0f} s")

    data = json.load(open(DATASET_PATH, encoding="utf-8"))
    casos = [(d, expected_pages(d["source"])) for d in data]
    casos = [(d, p) for d, p in casos if p and d["type"] in ("directa", "multi-documento")]
    print(f"🧪 Preguntas con página esperada: {len(casos)}\n")

    kmax = max(KS)
    hits = {k: 0 for k in KS}
    for i, (d, esperadas) in enumerate(casos, 1):
        recuperados = [doc_page(doc) for doc in vectordb.similarity_search(d["question"], k=kmax)]
        # Posición (1-based) del primer fragmento que cae en una página esperada
        pos = next((j for j, dp in enumerate(recuperados, 1) if dp in esperadas), None)
        for k in KS:
            if pos is not None and pos <= k:
                hits[k] += 1
        marca = "✅" if pos and pos <= min(KS) else ("🟡" if pos else "❌")
        print(f"{marca} [{i:02d}] ({d['type']}) {d['question'][:70]}")
        print(f"      esperado: {sorted(esperadas)}")
        print(f"      top-{min(KS)}:   {recuperados[:min(KS)]}")
        print(f"      primera coincidencia en posición: {pos if pos else f'no está en top-{kmax}'}")

    print("\n📊 RESUMEN")
    for k in KS:
        print(f"   hit@{k}: {hits[k]}/{len(casos)} = {hits[k] / len(casos):.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
