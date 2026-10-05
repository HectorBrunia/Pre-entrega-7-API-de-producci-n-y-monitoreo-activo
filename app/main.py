"""
main.py — API REST (FastAPI) del orquestador multi-agente.

    POST /tasks                 → crea el job en Redis (PENDING), lo encola en Arq y devuelve job_id (202)
    GET  /tasks/{job_id}        → estado del job (polling): PENDING/RUNNING/WAITING_APPROVAL/DONE/FAILED
    GET  /tasks                 → últimos jobs
    POST /tasks/{job_id}/approve→ decisión humana para un job en WAITING_APPROVAL (ver app/hitl.py)
    GET  /health                → Redis accesible

Ningún endpoint ejecuta el agente: todos son `async` y solo hacen operaciones cortas contra Redis con
el cliente asíncrono (redis.asyncio / arq), así el event loop nunca se bloquea por un LLM.

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app import jobs
from app.config import COLA, REDIS_URL
from app.hitl import router as router_hitl
from app.jobs import Estado
from app.observability import init_observabilidad

logger = logging.getLogger("app.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_observabilidad("api")
    app.state.redis = Redis.from_url(REDIS_URL, decode_responses=True)
    app.state.arq = await create_pool(RedisSettings.from_dsn(REDIS_URL), default_queue_name=COLA)
    # Abre las conexiones ahora y no en el primer request (con Redis remoto, cada handshake suma RTTs)
    await app.state.redis.ping()
    yield
    await app.state.arq.aclose()
    await app.state.redis.aclose()


app = FastAPI(
    title="API del orquestador multi-agente (TechCorp)",
    version="1.0.0",
    description="Ejecución asíncrona del sistema multi-agente del M6 con Redis, LangSmith y Human-in-the-loop.",
    lifespan=lifespan,
)
app.include_router(router_hitl)


class NuevaTarea(BaseModel):
    pregunta: str = Field(min_length=5, max_length=2000, examples=[
        "Un empleado va a cumplir 12 años en TechCorp. ¿Cuántos días de vacaciones acumuló en total "
        "según cada tramo de antigüedad (primeros 5 años, siguientes 5 y últimos 2)?"
    ])
    etiquetas: list[str] = Field(default_factory=list, max_length=10, description="Tags extra para filtrar la traza")
    simular_fallo: bool = Field(default=False, description="Solo demo: el worker lanza una excepción → FAILED")


@app.post("/tasks", status_code=status.HTTP_202_ACCEPTED)
async def crear_tarea(tarea: NuevaTarea, request: Request) -> dict:
    job_id = uuid.uuid4().hex
    r = request.app.state.redis
    await jobs.crear(r, job_id, tarea.pregunta, tarea.etiquetas, tarea.simular_fallo)
    try:
        await request.app.state.arq.enqueue_job("ejecutar_tarea", job_id, _job_id=job_id)
    except Exception as e:
        logger.exception("No se pudo encolar %s", job_id)
        await jobs.actualizar(r, job_id, status=Estado.FAILED, error=f"No se pudo encolar: {e}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "La cola no está disponible") from e
    return {"job_id": job_id, "status": Estado.PENDING, "links": {"estado": f"/tasks/{job_id}"}}


@app.get("/tasks/{job_id}")
async def estado_tarea(job_id: str, request: Request) -> dict:
    job = await jobs.obtener(request.app.state.redis, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")
    if job["status"] == Estado.WAITING_APPROVAL:
        job["links"] = {"aprobar": f"/tasks/{job_id}/approve"}
    return job


@app.get("/tasks")
async def listar_tareas(request: Request, limite: int = Query(20, ge=1, le=100)) -> list[dict]:
    return await jobs.listar(request.app.state.redis, limite)


@app.get("/health")
async def health(request: Request) -> dict:
    try:
        await request.app.state.redis.ping()
    except Exception as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Redis no responde: {e}") from e
    return {"status": "ok", "redis": "ok"}
