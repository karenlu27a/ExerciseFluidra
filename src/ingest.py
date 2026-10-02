from pathlib import Path
import re

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = BASE_DIR / "data" / "knowledgebase"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "university_policies"

EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_SIZE = 1024


def get_document_type(filename: str) -> str:
    name = filename.lower()
    if "calendario" in name:
        return "calendario"
    if "politica" in name:
        return "politica"
    if "reglamento" in name:
        return "reglamento"
    return "documento"


def get_year(text: str):
    match = re.search(r"\b(20\d{2})\b", text)
    return int(match.group(1)) if match else None


def chunk_document(text: str, filename: str):
    """
    Structure-aware chunking:
    - Regulations/policies: one Article = one semantic chunk.
    - Academic calendar: one academic period = one chunk.
    """
    doc_type = get_document_type(filename)
    year = get_year(text)

    if doc_type == "calendario":
        starts = list(re.finditer(
            r"(?=Periodo académico \d{4}-\d)",
            text,
            flags=re.IGNORECASE,
        ))

        chunks = []
        for i, match in enumerate(starts):
            start = match.start()
            end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
            chunk_text = text[start:end].strip()

            period_match = re.search(
                r"Periodo académico (\d{4}-\d)",
                chunk_text,
                flags=re.IGNORECASE,
            )
            period = period_match.group(1) if period_match else None

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "document": filename,
                    "document_type": doc_type,
                    "section": period or "Calendario académico",
                    "period": period,
                    "year": year,
                },
            })
        return chunks

    parts = re.split(r"(?=Artículo\s+\d+\.)", text, flags=re.IGNORECASE)
    chunks = []

    for part in parts:
        part = part.strip()
        if not re.match(r"Artículo\s+\d+\.", part, re.IGNORECASE):
            continue

        match = re.match(
            r"Artículo\s+(\d+)\.\s*(.+?)(?:\n|$)",
            part,
            flags=re.IGNORECASE,
        )

        article = int(match.group(1)) if match else None
        title = match.group(2).strip() if match else None

        chunks.append({
            "text": part,
            "metadata": {
                "document": filename,
                "document_type": doc_type,
                "section": title,
                "article": article,
                "year": year,
            },
        })

    return chunks


def recreate_collection(client: QdrantClient):
    collections = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME in collections:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )


def main():
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(f"No existe: {DOCUMENTS_DIR}")

    print(f"Cargando embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Conectando a Qdrant: {QDRANT_URL}")
    client = QdrantClient(path="./data/qdrant")
    # Falla temprano si Qdrant no está disponible.
    client.get_collections()

    recreate_collection(client)

    all_chunks = []

    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_document(text, path.name)
        print(f"{path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    texts = [chunk["text"] for chunk in all_chunks]

    print(f"Generando embeddings para {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    points = []

    for idx, (chunk, vector) in enumerate(zip(all_chunks, embeddings)):
        payload = {
            "text": chunk["text"],
            **chunk["metadata"],
        }

        points.append(
            models.PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload=payload,
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    info = client.get_collection(COLLECTION_NAME)

    print("\n=== INGESTA COMPLETADA ===")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vectors: {info.points_count}")
    print(f"Vector size: {VECTOR_SIZE}")
    print("Distance: COSINE")


if __name__ == "__main__":
    main()
