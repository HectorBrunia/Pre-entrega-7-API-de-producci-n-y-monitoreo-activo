"""
Tests de la API y del worker contra un Redis falso (fakeredis, con soporte de Lua).

- La API responde al instante y solo encola (el worker no se ejecuta en estos requests).
- El worker deja SIEMPRE un estado terminal: DONE, WAITING_APPROVAL o FAILED (nunca RUNNING colgado).
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app import jobs, worker
from app.graph import construir_grafo
from app.jobs import Estado
from app.main import app
from orquestador.state import Hallazgo
from tests.fakes import Guionado, decision, respuesta_analista, respuesta_investigador
from tests.test_hitl_grafo import EVIDENCIA_CAPACITACION, PREGUNTA_CAPACITACION, PREGUNTA_VACACIONES


class ColaFalsa:
    """Reemplaza al pool de Arq: registra lo encolado."""

    def __init__(self):
        self.encolados = []

    async def enqueue_job(self, funcion, *args, **kwargs):
        self.encolados.append((funcion, args))


@pytest_asyncio.fixture
async def redis():
    r = FakeAsyncRedis(decode_responses=True)
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def cliente(redis):
    app.state.redis = redis
    app.state.arq = ColaFalsa()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_post_encola_y_devuelve_job_id(cliente, redis):
    resp = await cliente.post("/tasks", json={"pregunta": PREGUNTA_VACACIONES, "etiquetas": ["test"]})

    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert app.state.arq.encolados == [("ejecutar_tarea", (job_id,))]

    estado = (await cliente.get(f"/tasks/{job_id}")).json()
    assert estado["status"] == "PENDING"
    assert estado["etiquetas"] == ["test"]


@pytest.mark.asyncio
async def test_job_inexistente_404(cliente):
    assert (await cliente.get("/tasks/nope")).status_code == 404
    assert (await cliente.post("/tasks/nope/approve", json={"aprobado": True})).status_code == 404


@pytest.mark.asyncio
async def test_pregunta_invalida_422(cliente):
    assert (await cliente.post("/tasks", json={"pregunta": "?"})).status_code == 422


@pytest.mark.asyncio
async def test_aprobar_solo_en_waiting_approval_y_una_sola_vez(cliente, redis):
    job_id = (await cliente.post("/tasks", json={"pregunta": PREGUNTA_CAPACITACION})).json()["job_id"]

    # Todavía PENDING → no se puede aprobar
    assert (await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True})).status_code == 409

    await jobs.actualizar(redis, job_id, status=Estado.WAITING_APPROVAL, aprobacion_pendiente={"motivos": ["x"]})
    resp = await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "revisor": "ana"})
    assert resp.status_code == 202
    assert app.state.arq.encolados[-1] == (
        "reanudar_tarea", (job_id, {"aprobado": True, "revisor": "ana", "comentario": None})
    )
    # Una segunda aprobación concurrente pierde el compare-and-set
    assert (await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": False})).status_code == 409


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class GrafoQueExplota:
    async def ainvoke(self, entrada, config):
        raise ConnectionError("Gemini no responde")


def _ctx(redis, grafo):
    return {"jobs": redis, "grafo": grafo}


def _grafo_critico():
    hallazgos = [
        Hallazgo(dato="Presupuesto anual de líderes", valor=1500, unidad="USD", fuente="politica_capacitacion.txt"),
        Hallazgo(dato="Años máximos de acumulación", valor=2, unidad="años", fuente="politica_capacitacion.txt"),
    ]
    return construir_grafo(
        investigador=Guionado(respuesta_investigador(hallazgos=hallazgos, evidencia=EVIDENCIA_CAPACITACION)),
        analista=Guionado(respuesta_analista(expresion="1500*2 - 2300", resultado=700.0)),
        router=Guionado(decision("investigador"), decision("analista"), decision("FINISH")),
        redactor=Guionado(AIMessage(content="Le sobran 700 USD.")),
        checkpointer=InMemorySaver(),
    )


@pytest.mark.asyncio
async def test_excepcion_en_el_worker_deja_el_job_en_failed(redis):
    await jobs.crear(redis, "j1", PREGUNTA_VACACIONES, [])

    resultado = await worker.ejecutar_tarea(_ctx(redis, GrafoQueExplota()), "j1")

    job = await jobs.obtener(redis, "j1")
    assert resultado == job["status"] == "FAILED"
    assert "ConnectionError: Gemini no responde" in job["error"]
    assert job["finished_at"] is not None


@pytest.mark.asyncio
async def test_simular_fallo_deja_el_job_en_failed(redis):
    await jobs.crear(redis, "j2", PREGUNTA_VACACIONES, [], simular_fallo=True)

    await worker.ejecutar_tarea(_ctx(redis, _grafo_critico()), "j2")

    job = await jobs.obtener(redis, "j2")
    assert job["status"] == "FAILED" and "FalloSimulado" in job["error"]


@pytest.mark.asyncio
async def test_timeout_deja_el_job_en_failed(redis, monkeypatch):
    import asyncio

    class GrafoLento:
        async def ainvoke(self, entrada, config):
            await asyncio.sleep(5)

    monkeypatch.setattr(worker, "TIMEOUT_EJECUCION_S", 0.05)
    await jobs.crear(redis, "j3", PREGUNTA_VACACIONES, [])

    await worker.ejecutar_tarea(_ctx(redis, GrafoLento()), "j3")

    job = await jobs.obtener(redis, "j3")
    assert job["status"] == "FAILED" and "TimeoutError" in job["error"]


@pytest.mark.asyncio
async def test_flujo_completo_hitl_waiting_approval_y_done(redis):
    ctx = _ctx(redis, _grafo_critico())
    await jobs.crear(redis, "j4", PREGUNTA_CAPACITACION, [])

    assert await worker.ejecutar_tarea(ctx, "j4") == "WAITING_APPROVAL"
    job = await jobs.obtener(redis, "j4")
    assert job["aprobacion_pendiente"]["propuesta"]["calculo"] == "1500*2 - 2300 = 700"

    assert await worker.reanudar_tarea(ctx, "j4", {"aprobado": True, "revisor": "ana"}) == "DONE"
    job = await jobs.obtener(redis, "j4")
    assert job["resultado"]["task_completed"] is True
    assert "Aprobado por revisor humano (ana)" in job["resultado"]["respuesta_final"]
    assert job["aprobacion_pendiente"] is None


@pytest.mark.asyncio
async def test_reaper_marca_failed_los_jobs_huerfanos(redis, monkeypatch):
    await jobs.crear(redis, "j5", PREGUNTA_VACACIONES, [])
    await jobs.actualizar(redis, "j5", status=Estado.RUNNING)
    monkeypatch.setattr(worker, "TIMEOUT_EJECUCION_S", -10_000)  # todo RUNNING cuenta como colgado

    await worker.reaper({"jobs": redis})

    job = await jobs.obtener(redis, "j5")
    assert job["status"] == "FAILED" and "huérfano" in job["error"]
