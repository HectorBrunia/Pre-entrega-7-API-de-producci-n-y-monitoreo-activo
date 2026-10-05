"""
Tests offline del orquestador del M6 (portados): camino feliz, bucle de reparación,
caminos de fallo y garantías del estado.

    python -m pytest -v
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

from app.graph import RECURSION_LIMIT, construir_grafo
from orquestador.state import ViolacionDeEscritura, estado_inicial, verificar_escritura
from tests.fakes import Guionado, decision, respuesta_analista, respuesta_investigador
from orquestador.tools.calculator import evaluar_expresion

PREGUNTA = (
    "Un empleado va a cumplir 12 años en TechCorp. Tuvo los días de vacaciones de cada tramo de "
    "antigüedad: primeros 5 años, siguientes 5 años y los últimos 2. ¿Cuántos días acumuló en total?"
)


def correr(investigador, analista, router, redactor=None):
    redactor = redactor or Guionado(AIMessage(content="Respuesta final."))
    app = construir_grafo(investigador=investigador, analista=analista, router=router, redactor=redactor)
    return asyncio.run(app.ainvoke(estado_inicial(PREGUNTA), {"recursion_limit": RECURSION_LIMIT}))


def recorrido(estado) -> list[str]:
    return [e.nodo for e in estado["bitacora"]]


# ---------------------------------------------------------------------------
# Camino feliz
# ---------------------------------------------------------------------------
def test_camino_feliz_investigador_analista_sintesis():
    router = Guionado(decision("investigador"), decision("analista"), decision("FINISH"))
    final = correr(Guionado(respuesta_investigador()), Guionado(respuesta_analista()), router)

    assert recorrido(final) == [
        "supervisor", "investigador", "validador",
        "supervisor", "analista", "validador",
        "supervisor", "sintesis",
    ]
    assert final["task_completed"] and not final["escalado_humano"]
    assert "5*14 + 5*21 + 2*28 = 231" in final["respuesta_final"]
    assert "politica_vacaciones.txt" in final["respuesta_final"]
    assert [c["agente"] for c in final["contribuciones"]] == ["investigador", "analista"]


def test_contexto_acotado_el_especialista_no_recibe_el_historial():
    investigador = Guionado(respuesta_investigador())
    router = Guionado(decision("investigador", instruccion="Buscá los tramos de vacaciones"), decision("analista"), decision("FINISH"))
    correr(investigador, Guionado(respuesta_analista()), router)

    mensajes = investigador.entradas[0]["messages"]
    assert len(mensajes) == 1
    assert mensajes[0].content == "Buscá los tramos de vacaciones"


# ---------------------------------------------------------------------------
# Bucle de reparación (el supervisor pide REFINAR)
# ---------------------------------------------------------------------------
def test_analista_con_resultado_incorrecto_es_reparado_con_error_especifico():
    analista = Guionado(respuesta_analista(resultado=245.0), respuesta_analista())
    router = Guionado(decision("investigador"), decision("analista"), decision("FINISH"))
    final = correr(Guionado(respuesta_investigador()), analista, router)

    assert final["intentos_analista"] == 2
    assert final["validaciones"][1].aprobado is False
    feedback = analista.entradas[1]["messages"][0].content
    assert "da 231, pero declaraste resultado=245" in feedback
    assert final["task_completed"]


def test_analista_que_usa_un_numero_sin_procedencia_es_rechazado():
    """Conflicto Analista vs Investigador: usa 30 días, que ningún hallazgo respalda."""
    analista = Guionado(respuesta_analista("5*14 + 5*21 + 2*30", 235.0), respuesta_analista())
    router = Guionado(decision("investigador"), decision("analista"), decision("FINISH"))
    final = correr(Guionado(respuesta_investigador()), analista, router)

    rechazo = final["validaciones"][1]
    assert not rechazo.aprobado and "['30']" in rechazo.errores[0]
    assert final["analysis_output"].resultado == 231.0


def test_investigador_que_alucina_un_valor_es_rechazado():
    from orquestador.state import Hallazgo

    alucinado = respuesta_investigador(
        hallazgos=[Hallazgo(dato="Vacaciones con más de 10 años", valor=35, unidad="días", fuente="politica_vacaciones.txt")]
    )
    router = Guionado(decision("investigador"), decision("analista"), decision("FINISH"))
    final = correr(Guionado(alucinado, respuesta_investigador()), Guionado(respuesta_analista()), router)

    assert "no aparece en el texto recuperado" in final["validaciones"][0].errores[0]
    assert final["intentos_investigador"] == 2 and final["task_completed"]


# ---------------------------------------------------------------------------
# Caminos de fallo
# ---------------------------------------------------------------------------
def test_especialista_que_lanza_excepcion_no_rompe_el_grafo_y_escala_a_humano():
    investigador = Guionado(RuntimeError("Pinecone timeout"))
    router = Guionado(decision("investigador"))
    final = correr(investigador, Guionado(respuesta_analista()), router)

    assert final["escalado_humano"] is True
    assert final["intentos_investigador"] == 3  # 1 intento + 2 reparaciones
    assert "Pinecone timeout" in final["respuesta_final"]
    assert recorrido(final)[-1] == "escalar_humano"


def test_supervisor_llm_caido_usa_politica_de_respaldo():
    router = Guionado(ConnectionError("Gemini 503"))
    final = correr(Guionado(respuesta_investigador()), Guionado(respuesta_analista()), router)

    assert final["task_completed"]
    assert "política de respaldo" in final["bitacora"][0].detalle
    assert final["analysis_output"].resultado == 231.0


def test_supervisor_infinito_se_corta_por_max_pasos():
    """El LLM insiste en re-investigar para siempre: MAX_PASOS lo corta."""
    router = Guionado(decision("investigador", razon="quiero más datos"))
    final = correr(Guionado(respuesta_investigador()), Guionado(respuesta_analista()), router)

    assert final["task_completed"]
    assert "máximo de" in [e for e in final["bitacora"] if e.nodo == "supervisor"][-1].detalle


def test_guardrail_no_permite_finish_sin_datos_validados():
    router = Guionado(decision("FINISH"), decision("analista"), decision("FINISH"))
    final = correr(Guionado(respuesta_investigador()), Guionado(respuesta_analista()), router)

    assert "OVERRIDE" in final["bitacora"][0].detalle
    assert recorrido(final)[1] == "investigador"


# ---------------------------------------------------------------------------
# Estado y herramientas
# ---------------------------------------------------------------------------
def test_aislamiento_de_escritura():
    with pytest.raises(ViolacionDeEscritura):
        verificar_escritura("analista", {"research_output": None})
    assert verificar_escritura("analista", {"analysis_output": None, "bitacora": []})


def test_confidence_score_fuera_de_rango_es_rechazado_por_pydantic():
    from pydantic import ValidationError

    from orquestador.state import ResearchFindings

    with pytest.raises(ValidationError):
        ResearchFindings(hallazgos=[], resumen="x", confidence_score=1.7)


@pytest.mark.parametrize("maliciosa", ["__import__('os').system('dir')", "open('x')", "2**10**10", "a+1"])
def test_calculadora_rechaza_codigo_arbitrario(maliciosa):
    with pytest.raises(Exception):
        evaluar_expresion(maliciosa)


def test_calculadora_aritmetica():
    assert evaluar_expresion("5*14 + 5*21 + 2*28") == 231
