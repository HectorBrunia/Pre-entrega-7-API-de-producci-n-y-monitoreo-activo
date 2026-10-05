"""
hitl.py — Human-in-the-loop: pausa obligatoria para aprobación humana en tareas críticas.

Dónde se intercala en el grafo:

    supervisor ──(FINISH)──▶ ¿es crítica? ──no──▶ sintesis ──▶ END
                                  │
                                  sí
                                  ▼
                          aprobacion_humana ── interrupt() ──  ⏸ el grafo se detiene; el checkpoint
                                  │                              queda en Redis y el job pasa a
                                  │                              WAITING_APPROVAL
              POST /tasks/{id}/approve ──▶ Command(resume=decisión) ──▶ el worker retoma el grafo
                                  │
                       aprobado ──┴── rechazado
                          ▼              ▼
                       sintesis      rechazado ──▶ END

Una tarea es CRÍTICA (criterio determinístico, sin LLM) si:
  1. Impacto económico: la respuesta va a informar montos de dinero: algún hallazgo validado está en
     USD/$/ARS/EUR, o la pregunta habla de dinero y hay un cálculo validado (ej: "la hora vale 12 USD,
     ¿cuánto cobra?"). Es la acción "con efectos secundarios": un monto mal informado se paga.
  2. Costo alto: la ejecución ya consumió muchos pasos del supervisor (≥ HITL_UMBRAL_PASOS), señal de
     que hubo reparaciones y de que la corrida es cara: alguien debería revisarla antes de entregarla.

La decisión humana queda en el estado (`aprobacion`), en la bitácora y en la traza de LangSmith.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Request, status
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app import jobs
from app.config import UMBRAL_PASOS_CRITICO
from app.jobs import Estado
from app.observability import traceable
from orquestador.agents.common import formatear_hallazgo
from orquestador.state import AgentState, EventoBitacora, aprobado

logger = logging.getLogger(__name__)

_MONEDA = re.compile(r"\b(usd|ars|eur|d[oó]lares?|pesos?)\b|\$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Lógica del grafo
# ---------------------------------------------------------------------------
@traceable(name="evaluar_criticidad", run_type="chain")
def evaluar_criticidad(state: AgentState) -> list[str]:
    """Motivos por los que la tarea requiere aprobación humana (lista vacía = no es crítica)."""
    motivos = []
    research = state.get("research_output")
    montos = []
    if research and aprobado(state, "investigador"):
        montos = [h for h in research.hallazgos if h.unidad and _MONEDA.search(h.unidad)]
    if montos:
        detalle = ", ".join(formatear_hallazgo(h) for h in montos)
        motivos.append(f"Impacto económico: la respuesta informa montos de dinero ({detalle}).")
    elif _MONEDA.search(state["tarea"]) and aprobado(state, "analista"):
        # El monto puede venir de la PREGUNTA (ej: "la hora vale 12 USD") y no de la política:
        # el resultado del cálculo igual es dinero a pagar.
        analisis = state["analysis_output"]
        motivos.append(
            f"Impacto económico: la pregunta involucra dinero y el cálculo da {analisis.resultado:g} "
            f"({analisis.expresion})."
        )
    pasos = state.get("pasos", 0)
    if pasos >= UMBRAL_PASOS_CRITICO:
        motivos.append(f"Costo alto: la ejecución usó {pasos} pasos del supervisor (umbral {UMBRAL_PASOS_CRITICO}).")
    return motivos


def propuesta_para_revisor(state: AgentState) -> dict:
    """Lo que el humano necesita ver para decidir: datos validados y cálculo propuesto."""
    research = state.get("research_output")
    analisis = state.get("analysis_output")
    return {
        "hallazgos": [f"{formatear_hallazgo(h)} [{h.fuente}]" for h in research.hallazgos] if research else [],
        "calculo": (
            f"{analisis.expresion} = {analisis.resultado:g}" if analisis and aprobado(state, "analista") else None
        ),
    }


async def nodo_aprobacion_humana(state: AgentState) -> dict:
    """Pausa el grafo con `interrupt()` hasta que llegue la decisión humana por la API.

    Al reanudar, LangGraph vuelve a ejecutar este nodo desde el principio y `interrupt()` devuelve
    el valor de `Command(resume=...)`. Por eso todo lo previo al interrupt es determinístico.
    """
    motivos = evaluar_criticidad(state)
    decision = interrupt(
        {
            "tipo": "aprobacion_requerida",
            "motivos": motivos,
            "pregunta": state["tarea"],
            "propuesta": propuesta_para_revisor(state),
        }
    )
    decision = decision if isinstance(decision, dict) else {"aprobado": bool(decision)}
    registro = {
        "aprobado": bool(decision.get("aprobado")),
        "revisor": decision.get("revisor"),
        "comentario": decision.get("comentario"),
        "motivos": motivos,
    }
    veredicto = "APROBADA" if registro["aprobado"] else "RECHAZADA"
    return {
        "aprobacion": registro,
        "bitacora": [
            EventoBitacora(
                nodo="aprobacion_humana",
                accion=f"tarea crítica {veredicto} por {registro['revisor'] or 'revisor anónimo'}",
                detalle=" | ".join(motivos) + (f" | comentario: {registro['comentario']}" if registro["comentario"] else ""),
            )
        ],
    }


def enrutar_aprobacion(state: AgentState) -> str:
    return "sintesis" if (state.get("aprobacion") or {}).get("aprobado") else "rechazado"


async def nodo_rechazado(state: AgentState) -> dict:
    aprobacion = state.get("aprobacion") or {}
    texto = (
        "⛔ RESPUESTA RECHAZADA EN LA REVISIÓN HUMANA\n"
        f"Pregunta: {state['tarea']}\n"
        f"Revisor: {aprobacion.get('revisor') or 'anónimo'}\n"
        f"Comentario: {aprobacion.get('comentario') or '(sin comentario)'}\n"
        "No se entrega la respuesta automática."
    )
    return {
        "respuesta_final": texto,
        "task_completed": False,
        "bitacora": [EventoBitacora(nodo="rechazado", accion="respuesta descartada por el revisor")],
    }


# ---------------------------------------------------------------------------
# Endpoint de aprobación
# ---------------------------------------------------------------------------
router = APIRouter(tags=["human-in-the-loop"])


class DecisionHumana(BaseModel):
    aprobado: bool = Field(description="true = se entrega la respuesta; false = se descarta")
    revisor: str | None = Field(default=None, max_length=100, description="Quién decide (queda en la auditoría)")
    comentario: str | None = Field(default=None, max_length=1000)


@router.post("/tasks/{job_id}/approve", status_code=status.HTTP_202_ACCEPTED)
async def aprobar_tarea(job_id: str, decision: DecisionHumana, request: Request) -> dict:
    """Registra la decisión humana y encola la reanudación del grafo (no la ejecuta acá)."""
    r = request.app.state.redis
    # Compare-and-set: solo un request puede sacar al job de WAITING_APPROVAL
    resultado = await jobs.transicion(
        r, job_id, {Estado.WAITING_APPROVAL}, Estado.PENDING, aprobacion=decision.model_dump()
    )
    if resultado == -1:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")
    if resultado == 0:
        job = await jobs.obtener(r, job_id)
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"El job está en estado {job['status']}: solo se puede aprobar/rechazar en WAITING_APPROVAL",
        )

    try:
        await request.app.state.arq.enqueue_job(
            "reanudar_tarea", job_id, decision.model_dump(), _job_id=f"{job_id}:reanudar"
        )
    except Exception as e:
        logger.exception("No se pudo encolar la reanudación de %s", job_id)
        await jobs.actualizar(r, job_id, status=Estado.FAILED, error=f"No se pudo encolar la reanudación: {e}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "La cola no está disponible") from e

    return {"job_id": job_id, "status": Estado.PENDING, "decision": decision.model_dump()}
