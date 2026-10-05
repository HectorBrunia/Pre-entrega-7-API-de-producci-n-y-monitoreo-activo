"""Dobles de prueba determinísticos: reemplazan a Gemini para testear el grafo sin API keys."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from orquestador.state import AnalysisResult, DecisionSupervisor, Hallazgo, ResearchFindings

EVIDENCIA_VACACIONES = (
    "[Fuente: politica_vacaciones.txt] Los empleados con menos de 5 años de antigüedad acceden a 14 días "
    "corridos de vacaciones por año. Los empleados con 5 a 10 años de antigüedad acceden a 21 días corridos "
    "por año. Los empleados con más de 10 años de antigüedad acceden a 28 días corridos por año."
)


class Guionado:
    """Devuelve respuestas en orden (repite la última). Si la respuesta es una excepción, la lanza."""

    def __init__(self, *respuestas):
        self.respuestas = list(respuestas)
        self.entradas: list = []

    async def ainvoke(self, entrada):
        self.entradas.append(entrada)
        r = self.respuestas.pop(0) if len(self.respuestas) > 1 else self.respuestas[0]
        if isinstance(r, Exception):
            raise r
        return r


def respuesta_investigador(hallazgos=None, evidencia=EVIDENCIA_VACACIONES, confianza=0.9):
    hallazgos = hallazgos or [
        Hallazgo(dato="Vacaciones con menos de 5 años", valor=14, unidad="días", fuente="politica_vacaciones.txt"),
        Hallazgo(dato="Vacaciones con 5 a 10 años", valor=21, unidad="días", fuente="politica_vacaciones.txt"),
        Hallazgo(dato="Vacaciones con más de 10 años", valor=28, unidad="días", fuente="politica_vacaciones.txt"),
    ]
    return {
        "messages": [ToolMessage(content=evidencia, name="buscar_en_politicas_internas", tool_call_id="t1")],
        "structured_response": ResearchFindings(hallazgos=hallazgos, resumen="Tramos de vacaciones", confidence_score=confianza),
    }


def respuesta_analista(expresion="5*14 + 5*21 + 2*28", resultado=231.0, confianza=0.95):
    return {
        "messages": [],
        "structured_response": AnalysisResult(
            expresion=expresion, resultado=resultado, explicacion="5 años a 14, 5 a 21 y 2 a 28", confidence_score=confianza
        ),
    }


def decision(next_, razon="test", instruccion="hacé tu parte"):
    return DecisionSupervisor(next=next_, razon=razon, instruccion=instruccion)


REDACTOR = Guionado(AIMessage(content="En 12 años acumuló 231 días de vacaciones."))
