"""
jobs.py — Estado de los trabajos persistido en Redis (fuente de verdad para el polling).

Cada job es un hash `job:{id}`; los campos complejos se guardan como JSON. Además, un sorted set
`jobs:recientes` (score = timestamp) permite listar los últimos jobs sin hacer SCAN.

Ciclo de vida:

    PENDING ──▶ RUNNING ──┬──▶ DONE
                          ├──▶ FAILED              (excepción, timeout o worker caído)
                          └──▶ WAITING_APPROVAL ──▶ PENDING (aprobado/rechazado) ──▶ RUNNING ──▶ …

Las transiciones que pueden competir entre sí (dos aprobaciones simultáneas del mismo job) se hacen
con un compare-and-set atómico en Lua: solo una gana.
"""

from __future__ import annotations

import json
import time
from enum import StrEnum
from typing import Any

from redis.asyncio import Redis

from app.config import TTL_JOB_S

PREFIJO = "job:"
INDICE = "jobs:recientes"


class Estado(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"


TERMINALES = {Estado.DONE, Estado.FAILED}
_CAMPOS_JSON = {"resultado", "aprobacion_pendiente", "aprobacion", "etiquetas"}

# KEYS[1] = hash del job · ARGV[1] = estados de origen permitidos ("A,B") · ARGV[2] = TTL ·
# ARGV[3..] = pares campo/valor a escribir. Devuelve 1 si aplicó, 0 si el estado no coincide, -1 si no existe.
_LUA_TRANSICION = """
local actual = redis.call('HGET', KEYS[1], 'status')
if not actual then return -1 end
if not string.find(',' .. ARGV[1] .. ',', ',' .. actual .. ',', 1, true) then return 0 end
redis.call('HSET', KEYS[1], unpack(ARGV, 3))
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
return 1
"""


def _ahora() -> float:
    return round(time.time(), 3)


def _clave(job_id: str) -> str:
    return PREFIJO + job_id


def _serializar(campos: dict[str, Any]) -> dict[str, str]:
    salida = {}
    for k, v in campos.items():
        if k in _CAMPOS_JSON:
            salida[k] = json.dumps(v, ensure_ascii=False, default=str)
        elif v is None:
            salida[k] = ""
        else:
            salida[k] = str(v)
    return salida


def _deserializar(datos: dict[str, str]) -> dict[str, Any]:
    job: dict[str, Any] = {}
    for k, v in datos.items():
        if k in _CAMPOS_JSON:
            job[k] = json.loads(v) if v else None
        elif k in {"created_at", "updated_at", "started_at", "finished_at"}:
            job[k] = float(v) if v else None
        elif k == "simular_fallo":
            job[k] = v == "True"
        else:
            job[k] = v or None
    return job


async def crear(r: Redis, job_id: str, pregunta: str, etiquetas: list[str], simular_fallo: bool = False) -> dict:
    ahora = _ahora()
    campos = {
        "job_id": job_id,
        "status": Estado.PENDING,
        "pregunta": pregunta,
        "etiquetas": etiquetas,
        "simular_fallo": simular_fallo,
        "created_at": ahora,
        "updated_at": ahora,
    }
    async with r.pipeline(transaction=True) as p:
        p.hset(_clave(job_id), mapping=_serializar(campos))
        p.expire(_clave(job_id), TTL_JOB_S)
        p.zadd(INDICE, {job_id: ahora})
        await p.execute()
    return await obtener(r, job_id)


async def obtener(r: Redis, job_id: str) -> dict | None:
    datos = await r.hgetall(_clave(job_id))
    return _deserializar(datos) if datos else None


async def actualizar(r: Redis, job_id: str, **campos: Any) -> None:
    campos["updated_at"] = _ahora()
    async with r.pipeline(transaction=True) as p:
        p.hset(_clave(job_id), mapping=_serializar(campos))
        p.expire(_clave(job_id), TTL_JOB_S)
        await p.execute()


async def transicion(r: Redis, job_id: str, desde: set[Estado], hacia: Estado, **campos: Any) -> int:
    """Compare-and-set atómico del estado. 1 = aplicada, 0 = estado distinto, -1 = no existe."""
    campos |= {"status": hacia, "updated_at": _ahora()}
    pares = [x for kv in _serializar(campos).items() for x in kv]
    origen = ",".join(sorted(desde))
    return int(await r.eval(_LUA_TRANSICION, 1, _clave(job_id), origen, TTL_JOB_S, *pares))


async def listar(r: Redis, limite: int = 20) -> list[dict]:
    ids = await r.zrevrange(INDICE, 0, limite - 1)
    jobs = []
    for job_id in ids:
        job = await obtener(r, job_id)
        if job is None:  # expiró: se limpia del índice
            await r.zrem(INDICE, job_id)
        else:
            jobs.append(job)
    return jobs


async def colgados(r: Redis, mas_viejos_que_s: float) -> list[str]:
    """Jobs en RUNNING sin actualizarse hace más de N segundos (el worker murió a mitad de camino)."""
    limite = _ahora() - mas_viejos_que_s
    ids = await r.zrangebyscore(INDICE, "-inf", limite)
    resultado = []
    for job_id in ids:
        job = await obtener(r, job_id)
        if job and job["status"] == Estado.RUNNING and (job.get("updated_at") or 0) < limite:
            resultado.append(job_id)
    return resultado
