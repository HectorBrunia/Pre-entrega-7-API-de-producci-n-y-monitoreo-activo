"""
observability.py — Instrumentación con LangSmith.

LangGraph/LangChain se auto-instrumentan cuando LANGSMITH_TRACING=true: cada nodo del grafo, cada
llamada a Gemini (con tokens de entrada/salida, de donde LangSmith calcula el costo) y cada tool
quedan como spans de la traza. Encima de eso:

- `@traceable` decora las funciones propias que no son runnables (la ejecución de un job y el
  chequeo de criticidad del HITL), así la traza raíz de cada job se llama igual y tiene su job_id.
- `metadata_de_traza()` agrega job_id, thread_id y etiquetas a cada traza: el `thread_id` hace que
  la ejecución inicial y la reanudación tras la aprobación humana aparezcan agrupadas en la
  vista "Threads" de LangSmith, y las etiquetas permiten filtrar una corrida de carga puntual.
"""

from __future__ import annotations

import logging
import os

from langsmith import traceable  # noqa: F401  (re-export: punto único de instrumentación)

logger = logging.getLogger(__name__)

PROYECTO_POR_DEFECTO = "api-agente-produccion"


def init_observabilidad(servicio: str) -> bool:
    """Valida la configuración de LangSmith. Devuelve True si las trazas están activas."""
    activo = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    if activo and not os.getenv("LANGSMITH_API_KEY"):
        logger.warning("LANGSMITH_TRACING=true pero falta LANGSMITH_API_KEY: trazas desactivadas")
        os.environ["LANGSMITH_TRACING"] = "false"
        activo = False
    if activo:
        os.environ.setdefault("LANGSMITH_PROJECT", PROYECTO_POR_DEFECTO)
        logger.info("[%s] Trazas → LangSmith, proyecto '%s'", servicio, os.environ["LANGSMITH_PROJECT"])
    else:
        logger.warning("[%s] Observabilidad desactivada (LANGSMITH_TRACING != true)", servicio)
    return activo


def metadata_de_traza(job_id: str, fase: str, etiquetas: list[str] | None = None) -> dict:
    """Metadata y tags que se adjuntan a la traza raíz (y heredan todos sus spans)."""
    return {
        "metadata": {
            "job_id": job_id,
            "thread_id": job_id,  # agrupa ejecución + reanudación en la vista Threads
            "fase": fase,
            "modelo": os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        },
        "tags": ["api-agente", f"fase:{fase}", *(etiquetas or [])],
    }


def vaciar_trazas() -> None:
    """Espera a que se envíen las trazas pendientes (al apagar el worker)."""
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception as e:  # nunca romper el apagado por la observabilidad
        logger.warning("No se pudieron vaciar las trazas: %s", e)
