"""
Tests del grafo con checkpointer + Human-in-the-loop (offline: dobles de prueba + InMemorySaver).

El InMemorySaver implementa la misma interfaz que el AsyncRedisSaver de producción, así que
interrupt()/Command(resume=...) se comportan igual.
"""

from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import RECURSION_LIMIT, construir_grafo
from orquestador.state import Hallazgo, estado_inicial
from tests.fakes import Guionado, decision, respuesta_analista, respuesta_investigador

PREGUNTA_VACACIONES = (
    "Un empleado va a cumplir 12 años en TechCorp. Tuvo los días de vacaciones de cada tramo de "
    "antigüedad: primeros 5 años, siguientes 5 años y los últimos 2. ¿Cuántos días acumuló en total?"
)
PREGUNTA_CAPACITACION = (
    "Un líder de equipo no usó su presupuesto de capacitación el año pasado. Este año quiere hacer un "
    "curso de 2300 USD. ¿Cuánto dinero le sobra o le falta?"
)
EVIDENCIA_CAPACITACION = (
    "[Fuente: politica_capacitacion.txt] Cada empleado dispone de un presupuesto anual de capacitación de "
    "800 USD. Los líderes de equipo disponen de 1500 USD anuales. El presupuesto no utilizado puede "
    "acumularse por un máximo de 2 años."
)


def _grafo_vacaciones():
    return construir_grafo(
        investigador=Guionado(respuesta_investigador()),
        analista=Guionado(respuesta_analista()),
        router=Guionado(decision("investigador"), decision("analista"), decision("FINISH")),
        redactor=Guionado(AIMessage(content="Acumuló 231 días.")),
        checkpointer=InMemorySaver(),
    )


def _grafo_capacitacion():
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


def _config():
    return {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": RECURSION_LIMIT}


@pytest.mark.asyncio
async def test_tarea_no_critica_termina_sin_pausa():
    grafo, config = _grafo_vacaciones(), _config()
    final = await grafo.ainvoke(estado_inicial(PREGUNTA_VACACIONES), config)

    snapshot = await grafo.aget_state(config)
    assert not snapshot.interrupts
    assert final["task_completed"]
    assert final["aprobacion"] is None
    assert "aprobacion_humana" not in [e.nodo for e in final["bitacora"]]


@pytest.mark.asyncio
async def test_tarea_critica_se_pausa_y_no_entrega_respuesta():
    grafo, config = _grafo_capacitacion(), _config()
    await grafo.ainvoke(estado_inicial(PREGUNTA_CAPACITACION), config)

    snapshot = await grafo.aget_state(config)
    assert snapshot.next == ("aprobacion_humana",)
    pausa = snapshot.interrupts[0].value
    assert pausa["tipo"] == "aprobacion_requerida"
    assert any("Impacto económico" in m for m in pausa["motivos"])
    assert pausa["propuesta"]["calculo"] == "1500*2 - 2300 = 700"
    assert snapshot.values["respuesta_final"] is None  # nada se entregó todavía


@pytest.mark.asyncio
async def test_aprobacion_reanuda_desde_el_checkpoint_y_sintetiza():
    grafo, config = _grafo_capacitacion(), _config()
    await grafo.ainvoke(estado_inicial(PREGUNTA_CAPACITACION), config)

    final = await grafo.ainvoke(Command(resume={"aprobado": True, "revisor": "rrhh", "comentario": "OK"}), config)

    assert final["task_completed"]
    assert final["aprobacion"]["aprobado"] is True
    assert "Aprobado por revisor humano (rrhh): OK" in final["respuesta_final"]
    nodos = [e.nodo for e in final["bitacora"]]
    assert nodos[-2:] == ["aprobacion_humana", "sintesis"]
    # No se re-ejecutó nada anterior a la pausa: un solo paso por cada especialista
    assert nodos.count("investigador") == 1 and nodos.count("analista") == 1


@pytest.mark.asyncio
async def test_rechazo_descarta_la_respuesta():
    grafo, config = _grafo_capacitacion(), _config()
    await grafo.ainvoke(estado_inicial(PREGUNTA_CAPACITACION), config)

    final = await grafo.ainvoke(Command(resume={"aprobado": False, "comentario": "monto incorrecto"}), config)

    assert not final["task_completed"]
    assert "RECHAZADA" in final["respuesta_final"]
    assert "monto incorrecto" in final["respuesta_final"]
    assert [e.nodo for e in final["bitacora"]][-1] == "rechazado"
