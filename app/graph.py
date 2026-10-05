"""
graph.py — Orquestador multi-agente del Módulo 6 + checkpointer de Redis + nodo Human-in-the-loop.

    START → supervisor ─┬─→ investigador ─→ validador ─→ supervisor
                        ├─→ analista ─────→ validador ─→ supervisor
                        ├─→ sintesis ─────────────────────────────→ END        (tarea no crítica)
                        ├─→ aprobacion_humana ⏸ ─┬─→ sintesis ────→ END        (crítica, aprobada)
                        │                         └─→ rechazado ───→ END        (crítica, rechazada)
                        └─→ escalar_humano ───────────────────────→ END

Cambios respecto del M6:
- `construir_grafo(checkpointer=...)`: con un checkpointer, cada super-step queda persistido en
  Redis bajo el `thread_id` del job. Eso es lo que permite PAUSAR en `interrupt()` y retomar más
  tarde, incluso desde otro proceso del worker.
- La arista condicional del supervisor desvía las tareas críticas al nodo `aprobacion_humana`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import wraps
from typing import AsyncIterator

from langgraph.graph import END, START, StateGraph

from app.config import TTL_CHECKPOINT_MIN
from app.hitl import enrutar_aprobacion, evaluar_criticidad, nodo_aprobacion_humana, nodo_rechazado
from orquestador.agents.analyst_agent import crear_nodo_analista
from orquestador.agents.research_agent import crear_nodo_investigador
from orquestador.agents.supervisor import crear_nodo_supervisor
from orquestador.agents.synthesizer import crear_nodo_sintesis, nodo_escalar_humano
from orquestador.state import MAX_PASOS, AgentState, DecisionSupervisor, DestinoRuteo, verificar_escritura
from orquestador.validation import nodo_validador

# Cada paso del supervisor dispara hasta 3 nodos (supervisor → especialista → validador),
# más la aprobación humana y la síntesis al final.
RECURSION_LIMIT = 3 * MAX_PASOS + 8


def _aislado(nombre: str, nodo):
    """Envuelve un nodo para que solo pueda escribir en su parcela del estado."""

    @wraps(nodo)
    async def envoltura(state: AgentState) -> dict:
        return verificar_escritura(nombre, await nodo(state))

    return envoltura


def enrutar_supervisor(state: AgentState) -> DestinoRuteo:
    """Arista condicional: la decisión del supervisor, salvo que finalizar requiera aprobación humana."""
    destino = state["next_agent"]
    if destino == "sintesis" and evaluar_criticidad(state):
        return "aprobacion_humana"
    return destino


def construir_grafo(investigador=None, analista=None, router=None, redactor=None, checkpointer=None):
    """Compila el grafo. Si no se pasan componentes, crea los reales con Gemini."""
    if None in (investigador, analista, router, redactor):
        from orquestador.agents.analyst_agent import crear_agente_analista
        from orquestador.agents.common import crear_llm
        from orquestador.agents.research_agent import crear_agente_investigador

        llm = crear_llm()
        investigador = investigador or crear_agente_investigador(llm)
        analista = analista or crear_agente_analista(llm)
        router = router or llm.with_structured_output(DecisionSupervisor)
        redactor = redactor or llm

    nodos = {
        "supervisor": crear_nodo_supervisor(router),
        "investigador": crear_nodo_investigador(investigador),
        "analista": crear_nodo_analista(analista),
        "validador": nodo_validador,
        "aprobacion_humana": nodo_aprobacion_humana,
        "rechazado": nodo_rechazado,
        "sintesis": crear_nodo_sintesis(redactor),
        "escalar_humano": nodo_escalar_humano,
    }

    grafo = StateGraph(AgentState)
    for nombre, nodo in nodos.items():
        grafo.add_node(nombre, _aislado(nombre, nodo))

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges(
        "supervisor",
        enrutar_supervisor,
        {d: d for d in ("investigador", "analista", "sintesis", "aprobacion_humana", "escalar_humano")},
    )
    grafo.add_edge("investigador", "validador")
    grafo.add_edge("analista", "validador")
    grafo.add_edge("validador", "supervisor")
    grafo.add_conditional_edges("aprobacion_humana", enrutar_aprobacion, {"sintesis": "sintesis", "rechazado": "rechazado"})
    grafo.add_edge("sintesis", END)
    grafo.add_edge("rechazado", END)
    grafo.add_edge("escalar_humano", END)

    return grafo.compile(checkpointer=checkpointer, name="orquestador_multiagente")


@asynccontextmanager
async def checkpointer_redis(redis_url: str) -> AsyncIterator:
    """AsyncRedisSaver listo para usar (crea los índices de RediSearch la primera vez).

    Requiere Redis 8 / Redis Stack (módulos JSON y Search); Redis Cloud free los incluye.
    Los checkpoints expiran a las TTL_CHECKPOINT_MIN para no llenar la base.
    """
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(
        redis_url, ttl={"default_ttl": TTL_CHECKPOINT_MIN, "refresh_on_read": True}
    ) as saver:
        await saver.asetup()
        yield saver
