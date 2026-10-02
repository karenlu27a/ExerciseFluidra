from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

QDRANT_DIR = BASE_DIR / "data" / "qdrant"

COLLECTION_NAME = "university_policies"

EMBEDDING_MODEL = "BAAI/bge-m3"

GENERATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TOP_K = 3

# Umbral mínimo para considerar que encontramos información relevante
MIN_SCORE = 0.35


# ============================================================
# CARGAR MODELOS
# ============================================================

def load_models():

    print("Cargando modelo de embeddings...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Cargando modelo generativo...")
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL,
        torch_dtype=torch.float32
    )

    return embedding_model, tokenizer, model


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_information(question, embedding_model, client):

    # Convertimos la pregunta en un vector
    query_vector = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # Buscamos los chunks más similares
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K
    ).points

    return results


# ============================================================
# CONSTRUIR CONTEXTO
# ============================================================

def build_context(results):

    context = ""

    for i, result in enumerate(results, start=1):

        payload = result.payload

        context += f"""
FUENTE {i}
Documento: {payload['document']}
Tipo: {payload['document_type']}
"""

        if "article" in payload:
            context += f"Artículo: {payload['article']}\n"

        if "period" in payload:
            context += f"Periodo: {payload['period']}\n"

        context += f"""
Contenido:
{payload['text']}

-------------------------
"""

    return context


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    question,
    context,
    tokenizer,
    model
):

    system_prompt = """
Eres un asistente virtual de la Universidad Virtual Andina.

Tu tarea es responder preguntas utilizando ÚNICAMENTE
la información proporcionada en el contexto.

No inventes información.

Si la respuesta no está en el contexto, responde exactamente:

"No encontré información suficiente en la base de conocimiento
de la universidad para responder esta pregunta."

Responde de manera clara, breve y natural.

No menciones que eres un modelo de lenguaje.
"""

    user_prompt = f"""
CONTEXTO RECUPERADO:

{context}

PREGUNTA DEL USUARIO:

{question}

RESPUESTA:
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=True
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# MOSTRAR FUENTES
# ============================================================

def show_sources(results):

    print("\n" + "=" * 60)
    print("INFORMACIÓN RECUPERADA")
    print("=" * 60)

    for i, result in enumerate(results, start=1):

        payload = result.payload

        print(f"\nFuente {i}")
        print(f"Score de similitud: {result.score:.3f}")
        print(f"Documento: {payload['document']}")
        print(f"Tipo: {payload['document_type']}")

        if "article" in payload:
            print(f"Artículo: {payload['article']}")

        if "period" in payload:
            print(f"Periodo: {payload['period']}")

        print(f"Contenido recuperado:")
        print(payload["text"])


# ============================================================
# CHATBOT
# ============================================================

def main():

    print("=" * 60)
    print("ASISTENTE VIRTUAL - UNIVERSIDAD VIRTUAL")
    print("=" * 60)

    print("\nInicializando sistema...\n")

    # Conectar con Qdrant
    client = QdrantClient(
        path=str(QDRANT_DIR)
    )

    if not client.collection_exists(COLLECTION_NAME):

        print(
            "No existe la base vectorial."
        )

        print(
            "\nPrimero ejecuta:"
        )

        print(
            "python src/ingest.py"
        )

        return

    # Cargar modelos
    embedding_model, tokenizer, generation_model = load_models()

    print("\nBot listo.")
    print("Escribe 'salir' para terminar.\n")

    while True:

        question = input("Tú: ").strip()

        if question.lower() == "salir":

            print("\nBot: ¡Hasta luego!")
            break

        if not question:
            continue

        # ----------------------------------------------------
        # 1. RETRIEVAL
        # ----------------------------------------------------

        results = retrieve_information(
            question,
            embedding_model,
            client
        )

        # ----------------------------------------------------
        # VALIDAR SI ENCONTRAMOS INFORMACIÓN
        # ----------------------------------------------------

        if not results or results[0].score < MIN_SCORE:

            print(
                "\nBot: No encontré información suficiente "
                "en la base de conocimiento de la universidad "
                "para responder esta pregunta."
            )

            continue

        # ----------------------------------------------------
        # MOSTRAR RETRIEVAL
        # ----------------------------------------------------

        show_sources(results)

        # ----------------------------------------------------
        # 2. AUGMENTATION
        # ----------------------------------------------------

        context = build_context(results)

        # ----------------------------------------------------
        # 3. GENERATION
        # ----------------------------------------------------

        answer = generate_answer(
            question,
            context,
            tokenizer,
            generation_model
        )

        # ----------------------------------------------------
        # RESPUESTA FINAL
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("RESPUESTA")
        print("=" * 60)

        print(f"\n{answer}")

        # ----------------------------------------------------
        # MOSTRAR TRAZABILIDAD
        # ----------------------------------------------------

        print("\n" + "-" * 60)
        print("RECUPERADO DE DOCUMENTO:")

        for result in results:

            payload = result.payload

            source = payload["document"]

            metadata = ""

            if "article" in payload:
                metadata = f" | Artículo {payload['article']}"

            if "period" in payload:
                metadata = f" | Periodo {payload['period']}"

            print(
                f"- {source}{metadata}"
                f" | similarity: {result.score:.3f}"
            )

        print("-" * 60)
        print()


if __name__ == "__main__":
    main()