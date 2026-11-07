# app/run_eval.py
"""
Evaluación del RAG con mlflow.genai.evaluate y jueces LLM (LLM-as-a-judge).

Cada pregunta del dataset genera una traza; los jueces la califican y dejan su
razonamiento (rationale) adjunto a la traza. El resultado agregado queda en un
run del experimento "rag_evaluation", enlazado a la versión de la app
(proveedor + modelo + prompt + k) y a la versión del prompt.

Uso:
    python app/run_eval.py
    PROMPT_VERSION=prompts:/asistente_calidad@v2_resumido_directo python app/run_eval.py
    RETRIEVER_K=8 python app/run_eval.py
    EVAL_SCORERS=correctness,retrieval_groundedness python app/run_eval.py   # solo algunos jueces
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

import mlflow
from dotenv import load_dotenv
from mlflow.genai.scorers import (
    Correctness,
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    Safety,
)

from app.llm_provider import describe, get_judge_model_uri
from app.prompt_registry import DEFAULT_PROMPT
from app.rag_pipeline import build_chain, load_vectorstore_from_disk
from app.tracking import EVAL_EXPERIMENT, app_config, init_tracking

load_dotenv()

# Configuración
PROMPT_VERSION = os.getenv("PROMPT_VERSION") or DEFAULT_PROMPT
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 1000)        # deben coincidir con el índice construido
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 150)
K = int(os.getenv("RETRIEVER_K") or 4)
DATASET_PATH = "tests/eval_dataset.json"

# Paralelismo de MLflow. Por defecto evalúa ~10 filas x 10 jueces a la vez, lo que
# supera los límites gratuitos de Groq (30 solicitudes y 8K tokens por minuto).
# Con Groq usamos un modo pausado; cualquier valor puesto en .env tiene prioridad.
if os.getenv("LLM_PROVIDER", "openai").strip().lower() == "groq":
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")         # una pregunta a la vez
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "2")  # 2 jueces en paralelo
    os.environ.setdefault("MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT", "0.2") # ~12 llamadas/min
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_RETRIES", "8")         # más reintentos ante 429


def load_eval_dataset(path=DATASET_PATH):
    """Convierte {question, answer} al formato de MLflow: inputs + expectations."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {
            "inputs": {"question": item["question"]},
            "expectations": {"expected_response": item["answer"]},
        }
        for item in raw
    ]


def get_scorers(judge_model: str):
    """
    Jueces predefinidos de MLflow + guías propias en lenguaje natural.
    👉 Desafío: agrega aquí tus propios scorers (ver README).
    Con EVAL_SCORERS=correctness,safety se ejecutan solo los indicados.
    """
    scorers = [
        Correctness(model=judge_model),            # ¿coincide con la respuesta esperada?
        RelevanceToQuery(model=judge_model),       # ¿responde a lo que se preguntó?
        RetrievalGroundedness(model=judge_model),  # ¿está sustentada en los documentos? (alucinación)
        RetrievalRelevance(model=judge_model),     # ¿los fragmentos recuperados son pertinentes?
        Safety(model=judge_model),                 # ¿contenido dañino u ofensivo?
        Guidelines(
            name="idioma_espanol",
            guidelines="La respuesta debe estar escrita en español.",
            model=judge_model,
        ),
        Guidelines(
            name="tono_profesional",
            guidelines=(
                "La respuesta debe tener un tono técnico y profesional, apropiado "
                "para un asistente de calidad de proveedores automotrices, y no debe "
                "inventar requisitos, plazos ni valores numéricos."
            ),
            model=judge_model,
        ),
    ]
    selected = [n.strip() for n in os.getenv("EVAL_SCORERS", "").split(",") if n.strip()]
    if selected:
        unknown = set(selected) - {sc.name for sc in scorers}
        if unknown:
            raise ValueError(f"EVAL_SCORERS desconocidos: {unknown}. "
                             f"Disponibles: {[sc.name for sc in scorers]}")
        scorers = [sc for sc in scorers if sc.name in selected]
    return scorers


def main():
    vectordb = load_vectorstore_from_disk()
    chain = build_chain(vectordb, prompt_version=PROMPT_VERSION, k=K)
    prompt_uri = chain.combine_docs_chain.llm_chain.prompt.metadata["prompt_uri"]

    model_id = init_tracking(EVAL_EXPERIMENT, prompt_uri, K)
    judge_model = get_judge_model_uri()

    print(f"📊 Experimento MLflow: {EVAL_EXPERIMENT}")
    print(f"⚙️  App: {describe()} | prompt={prompt_uri} | k={K}")
    scorers = get_scorers(judge_model)
    print(f"⚖️  Juez: {judge_model} | {len(scorers)} jueces: {[sc.name for sc in scorers]}")
    if os.getenv("MLFLOW_GENAI_EVAL_MAX_WORKERS") == "1":
        print("🐢 Modo pausado para respetar los límites del proveedor (puede tardar varios minutos)")

    def predict_fn(question: str) -> str:
        result = chain.invoke({"question": question, "chat_history": []})
        return result["answer"]

    run_name = f"{describe()['llm_provider']}|{prompt_uri.replace('prompts:/', '')}|k{K}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **app_config(prompt_uri, K),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "judge_model": judge_model,
            "dataset": DATASET_PATH,
        })
        results = mlflow.genai.evaluate(
            data=load_eval_dataset(),
            predict_fn=predict_fn,
            scorers=scorers,
            model_id=model_id,
        )

    print("\n📈 Resultados agregados:")
    for name, value in sorted(results.metrics.items()):
        print(f"   {name:40s} {value:.2f}" if isinstance(value, (int, float)) else f"   {name}: {value}")
    try:
        print_detalle_por_pregunta(results.run_id)
    except Exception as e:  # el detalle es solo diagnóstico: nunca debe romper la evaluación
        print(f"⚠️  No se pudo imprimir el detalle por pregunta: {e!r}")
    print(f"\n🔎 Detalle por pregunta (trazas + razonamiento de cada juez): mlflow ui → "
          f"experimento '{EVAL_EXPERIMENT}' → run '{run_name}'")
    return results


def _assessment_value(assessments, name):
    """Busca el veredicto y el razonamiento de un juez en la lista de assessments de una traza."""
    for a in assessments or []:
        a_name = getattr(a, "name", None) or (a.get("assessment_name") if isinstance(a, dict) else None)
        if a_name != name:
            continue
        if isinstance(a, dict):
            fb = a.get("feedback") or {}
            return fb.get("value"), a.get("rationale") or ""
        return getattr(a, "value", None), getattr(a, "rationale", "") or ""
    return None, ""


def _es_positivo(value):
    return str(value).strip().lower() in ("yes", "true", "1", "1.0", "pass")


def print_detalle_por_pregunta(run_id, metric="correctness"):
    """Imprime, para cada pregunta, el veredicto del juez, la respuesta y el porqué."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        tipos = {item["question"]: item.get("type", "?") for item in json.load(f)}

    traces = mlflow.search_traces(run_id=run_id)
    print(f"\n🧾 Detalle por pregunta ({metric}) — {len(traces)} trazas")
    por_tipo = {}
    for i, row in enumerate(traces.itertuples(index=False), 1):
        request = row.request if isinstance(row.request, dict) else json.loads(row.request or "{}")
        question = request.get("question", str(request))
        response = row.response if isinstance(row.response, str) else json.dumps(row.response, ensure_ascii=False)
        value, rationale = _assessment_value(row.assessments, metric)
        ok = _es_positivo(value)
        tipo = tipos.get(question, "?")
        por_tipo.setdefault(tipo, []).append(ok)
        print(f"{'✅' if ok else '❌'} [{i:02d}] ({tipo}) {question[:80]}")
        print(f"      respuesta: {(response or '').strip()[:220]!r}")
        if not ok:
            print(f"      juez:      {rationale.strip()[:300]!r}")

    print(f"\n📊 {metric} por tipo de pregunta:")
    for tipo, oks in sorted(por_tipo.items()):
        print(f"   {tipo:16s} {sum(oks)}/{len(oks)} = {sum(oks) / len(oks):.2f}")


if __name__ == "__main__":
    main()
