"""
worker.py — Worker de Arq: ejecuta el grafo en segundo plano, en un proceso separado de la API.

    arq app.worker.WorkerSettings

La API solo encola (`enqueue_job`) y responde al instante; este proceso consume la cola de Redis y
hace el trabajo pesado (llamadas a Gemini, RAG, validaciones). Así el event loop de FastAPI nunca se
bloquea, y se puede escalar levantando más workers.

Garantía de estado: TODA ejecución termina en DONE, WAITING_APPROVAL o FAILED.
  - Cualquier excepción del grafo → FAILED con el error (el cliente deja de hacer polling).
  - Timeout propio (`asyncio.timeout`) → FAILED. Se usa uno propio porque el timeout de Arq cancela
    la corrutina con CancelledError, que no es Exception y no pasaría por nuestro `except`.
  - Worker muerto a mitad de un job → la tarea periódica `reaper` lo marca FAILED.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import AsyncExitStack

from arq import cron
from arq.connections import RedisSettings
from langgraph.types import Command
from redis.asyncio import Redis

from app import jobs
from app.config import COLA, MAX_JOBS_CONCURRENTES, REDIS_URL, TIMEOUT_EJECUCION_S
from app.graph import RECURSION_LIMIT, checkpointer_redis, construir_grafo
from app.jobs import Estado
from app.observability import init_observabilidad, metadata_de_traza, traceable, vaciar_trazas
from orquestador.agents.common import describir_error
from orquestador.state import estado_inicial

logger = logging.getLogger("app.worker")


class FalloSimulado(RuntimeError):
    """Excepción inyectada a pedido (simular_fallo=true) para demostrar el camino a FAILED."""


# ---------------------------------------------------------------------------
# Ciclo de vida del worker
# ---------------------------------------------------------------------------
async def startup(ctx: dict) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    init_observabilidad("worker")

    pila = AsyncExitStack()
    ctx["pila"] = pila
    ctx["jobs"] = Redis.from_url(REDIS_URL, decode_responses=True)
    saver = await pila.enter_async_context(checkpointer_redis(REDIS_URL))
    ctx["grafo"] = construir_grafo(checkpointer=saver)

    # Pre-calienta el índice del RAG (BM25 / Pinecone) para que el primer job no pague ese costo
    from orquestador.tools.knowledge_base import obtener_retriever

    await asyncio.to_thread(obtener_retriever)
    logger.info("Worker listo: grafo compilado con AsyncRedisSaver, máx. %d jobs concurrentes", MAX_JOBS_CONCURRENTES)


async def shutdown(ctx: dict) -> None:
    await ctx["pila"].aclose()
    await ctx["jobs"].aclose()
    await asyncio.to_thread(vaciar_trazas)


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
@traceable(name="tarea_agente", run_type="chain")
async def _invocar_grafo(grafo, entrada, config: dict, simular_fallo: bool = False) -> dict:
    """Traza raíz de cada ejecución: contiene todos los nodos del grafo y las llamadas a Gemini."""
    if simular_fallo:
        raise FalloSimulado("fallo inyectado en el worker para demostrar el estado FAILED")
    return await grafo.ainvoke(entrada, config)


def _resumen_final(valores: dict) -> dict:
    analisis = valores.get("analysis_output")
    return {
        "respuesta_final": valores.get("respuesta_final"),
        "task_completed": valores.get("task_completed"),
        "escalado_humano": valores.get("escalado_humano"),
        "pasos": valores.get("pasos"),
        "calculo": f"{analisis.expresion} = {analisis.resultado:g}" if analisis and not analisis.error else None,
        "aprobacion": valores.get("aprobacion"),
        "bitacora": [e.model_dump() for e in valores.get("bitacora") or []],
    }


async def _correr(ctx: dict, job_id: str, entrada, fase: str) -> str:
    r: Redis = ctx["jobs"]
    job = await jobs.obtener(r, job_id)
    if job is None:
        logger.error("Job %s inexistente (¿expiró?)", job_id)
        return "missing"

    await jobs.actualizar(r, job_id, status=Estado.RUNNING, started_at=time.time(), error=None)
    traza = metadata_de_traza(job_id, fase, job.get("etiquetas"))
    config = {
        "configurable": {"thread_id": job_id},
        "recursion_limit": RECURSION_LIMIT,
        "metadata": traza["metadata"],
        "tags": traza["tags"],
    }
    grafo = ctx["grafo"]
    try:
        async with asyncio.timeout(TIMEOUT_EJECUCION_S):
            await _invocar_grafo(
                grafo, entrada, config, simular_fallo=bool(job.get("simular_fallo")),
                langsmith_extra={"metadata": traza["metadata"], "tags": traza["tags"]},
            )
        snapshot = await grafo.aget_state({"configurable": {"thread_id": job_id}})
    except TimeoutError:
        error = f"TimeoutError: la ejecución superó {TIMEOUT_EJECUCION_S}s"
        logger.error("Job %s: %s", job_id, error)
        await jobs.actualizar(r, job_id, status=Estado.FAILED, error=error, finished_at=time.time())
        return Estado.FAILED
    except Exception as e:
        logger.exception("Job %s falló", job_id)
        await jobs.actualizar(r, job_id, status=Estado.FAILED, error=describir_error(e), finished_at=time.time())
        return Estado.FAILED

    if snapshot.interrupts:
        await jobs.actualizar(
            r, job_id, status=Estado.WAITING_APPROVAL, aprobacion_pendiente=snapshot.interrupts[0].value
        )
        logger.info("Job %s en pausa esperando aprobación humana", job_id)
        return Estado.WAITING_APPROVAL

    await jobs.actualizar(
        r, job_id, status=Estado.DONE, resultado=_resumen_final(snapshot.values),
        aprobacion_pendiente=None, finished_at=time.time(),
    )
    logger.info("Job %s terminado", job_id)
    return Estado.DONE


async def ejecutar_tarea(ctx: dict, job_id: str) -> str:
    """Primera ejecución del grafo para un job recién creado."""
    job = await jobs.obtener(ctx["jobs"], job_id)
    if job is None:
        return "missing"
    return await _correr(ctx, job_id, estado_inicial(job["pregunta"]), fase="ejecucion")


async def reanudar_tarea(ctx: dict, job_id: str, decision: dict) -> str:
    """Retoma el grafo desde el checkpoint en Redis, entregándole la decisión humana al interrupt()."""
    return await _correr(ctx, job_id, Command(resume=decision), fase="reanudacion")


async def reaper(ctx: dict) -> None:
    """Marca FAILED los jobs que quedaron en RUNNING porque el worker murió sin terminarlos."""
    for job_id in await jobs.colgados(ctx["jobs"], TIMEOUT_EJECUCION_S + 120):
        aplicado = await jobs.transicion(
            ctx["jobs"], job_id, {Estado.RUNNING}, Estado.FAILED,
            error="El worker se detuvo durante la ejecución (job huérfano)", finished_at=time.time(),
        )
        if aplicado == 1:
            logger.warning("Job huérfano %s marcado como FAILED", job_id)


class WorkerSettings:
    functions = [ejecutar_tarea, reanudar_tarea]
    cron_jobs = [cron(reaper, minute=set(range(60)), run_at_startup=True)]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    queue_name = COLA
    max_jobs = MAX_JOBS_CONCURRENTES
    job_timeout = TIMEOUT_EJECUCION_S + 60  # red de seguridad; el corte real es asyncio.timeout
    max_tries = 1  # un LLM que falló no se reintenta a ciegas: el job queda FAILED y visible
