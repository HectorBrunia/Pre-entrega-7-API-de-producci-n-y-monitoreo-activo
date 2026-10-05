"""
Regresión: los modelos Pydantic del estado deben sobrevivir al serializador del AsyncRedisSaver.

Sin la allowlist de app/graph.py, al leer un checkpoint de Redis (al reanudar tras el HITL) los
modelos volvían como dict y el grafo fallaba con "'dict' object has no attribute 'agente'".
"""

from __future__ import annotations

from app.graph import _serializador_del_estado
from orquestador.state import AnalysisArtifact, EventoBitacora, Hallazgo, ResearchArtifact, ValidationReport


def _ida_y_vuelta(obj):
    serde = _serializador_del_estado()
    return serde.loads_typed(serde.dumps_typed(obj))


def test_modelos_del_estado_vuelven_con_su_tipo():
    estado = {
        "research_output": ResearchArtifact(
            hallazgos=[Hallazgo(dato="Presupuesto líderes", valor=1500, unidad="USD", fuente="politica_capacitacion.txt")],
            resumen="ok", confidence_score=0.9, fuentes_recuperadas=["politica_capacitacion.txt"], evidencia="1500 USD",
        ),
        "analysis_output": AnalysisArtifact(expresion="1500*2 - 2300", resultado=700, explicacion="x", confidence_score=0.9),
        "validaciones": [ValidationReport(agente="investigador", aprobado=True)],
        "bitacora": [EventoBitacora(nodo="supervisor", accion="paso 1")],
    }

    vuelta = _ida_y_vuelta(estado)

    assert isinstance(vuelta["research_output"], ResearchArtifact)
    assert isinstance(vuelta["research_output"].hallazgos[0], Hallazgo)
    assert vuelta["research_output"].hallazgos[0].valor == 1500
    assert isinstance(vuelta["analysis_output"], AnalysisArtifact)
    assert vuelta["validaciones"][0].agente == "investigador"
    assert vuelta["bitacora"][0].nodo == "supervisor"


def test_artefacto_de_error_sin_validar_tambien_se_reconstruye():
    # model_construct con hallazgos=[] viola min_length=1: el constructor normal lo rechazaría
    error = ResearchArtifact.model_construct(
        hallazgos=[], resumen="", confidence_score=0.0, fuentes_recuperadas=[], evidencia="", error="TimeoutError: x"
    )

    vuelta = _ida_y_vuelta({"research_output": error})

    assert isinstance(vuelta["research_output"], ResearchArtifact)
    assert vuelta["research_output"].error == "TimeoutError: x"
