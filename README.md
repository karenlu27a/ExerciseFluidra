# RAG PoC — Ingesta

Primera etapa del sistema RAG:

**TXT → chunking estructural → BGE-M3 → Qdrant**

## Requisitos

- Python 3.11 recomendado
- Docker Desktop
- Docker Compose
- Internet solamente para descargar las dependencias y el modelo BGE-M3 en la primera ejecución.

No necesitas API keys para esta etapa.

## 1. Preparar Python

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Levantar Qdrant

Con Docker Desktop abierto:

```bash
docker compose up -d
```

Qdrant quedará disponible en:

- http://localhost:6333
- Dashboard: http://localhost:6333/dashboard

## 3. Ejecutar la ingesta

Con el entorno virtual activado:

```bash
python src/ingest.py
```

La primera ejecución descarga `BAAI/bge-m3` desde Hugging Face. Después se reutiliza desde la caché local.

## ¿Qué hace `ingest.py`?

1. Lee los cinco `.txt`.
2. Hace chunking estructural:
   - reglamentos/políticas → un artículo por chunk;
   - calendario → un periodo académico por chunk.
3. Añade metadata.
4. Genera embeddings con BGE-M3.
5. Crea una colección Qdrant de 1024 dimensiones.
6. Usa distancia cosine.
7. Guarda vector + texto + metadata.

## Importante

El script recrea la colección en cada ejecución para que la PoC sea reproducible y no acumule datos duplicados.

Todavía NO implementa:
- Retrieval
- Augmentation
- LLM / Generation
- interfaz de usuario

Esas serán las siguientes etapas.
