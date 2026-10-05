"""
Agente de Análisis/Cómputo.

- Herramienta acotada: solo una calculadora segura (ast), sin acceso a la base de datos.
- Salida estructurada: devuelve un `AnalysisResult` (Pydantic) con la expresión exacta
  usada, para que el validador pueda RECALCULARLA de forma determinística.
- Contexto acotado: recibe la instrucción del supervisor + los hallazgos VALIDADOS del
  investigador (solo los datos, no la conversación).
- Escribe SOLO en su parcela: `analysis_output` e `intentos_analista`.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from orquestador.agents.common import describir_error, formatear_hallazgo
from orquestador.agents.prompts import ANALISTA_PROMPT
from orquestador.state import AgentState, AnalysisArtifact, AnalysisResult, EventoBitacora
from orquestador.tools.calculator import evaluar_expresion


@tool
def calculadora(expresion: str) -> str:
    """Evalúa una expresión aritmética (+, -, *, /, **, %, paréntesis). Ej: '5*14 + 5*21 + 2*28'."""
    try:
        return f"Resultado: {evaluar_expresion(expresion)}"
    except Exception as e:
        return f"Error al calcular '{expresion}': {e}"


def crear_agente_analista(llm):
    return create_agent(
        model=llm,
        tools=[calculadora],
        system_prompt=ANALISTA_PROMPT,
        response_format=ToolStrategy(AnalysisResult),
        name="analista",
    )


def armar_tarea(state: AgentState) -> str:
    partes = [f"PREGUNTA ORIGINAL: {state['tarea']}", f"INSTRUCCIÓN: {state['instruccion'] or 'Calculá lo que pide la pregunta.'}"]
    research = state.get("research_output")
    if research and research.hallazgos:
        datos = "\n".join(
            f"- {formatear_hallazgo(h)} [fuente: {h.fuente}]"
            for h in research.hallazgos
        )
        partes.append(f"DATOS VALIDADOS:\n{datos}")
    if state.get("feedback_reparacion"):
        partes.append(state["feedback_reparacion"])
    return "\n\n".join(partes)


def crear_nodo_analista(agente):
    async def nodo_analista(state: AgentState) -> dict:
        intento = state.get("intentos_analista", 0) + 1
        try:
            resultado = await agente.ainvoke({"messages": [HumanMessage(content=armar_tarea(state))]})
            analisis: AnalysisResult = resultado["structured_response"]
            artefacto = AnalysisArtifact(**analisis.model_dump())
            resumen = f"{artefacto.expresion} = {artefacto.resultado:g}. {artefacto.explicacion}"
            evento = EventoBitacora(
                nodo="analista",
                accion=f"intento {intento}: {artefacto.expresion} = {artefacto.resultado:g}",
                detalle=f"confianza={artefacto.confidence_score:.2f}",
            )
        except Exception as e:
            artefacto = AnalysisArtifact.model_construct(
                expresion="", resultado=0.0, explicacion="", confidence_score=0.0, error=describir_error(e)
            )
            resumen = f"ERROR: {artefacto.error}"
            evento = EventoBitacora(nodo="analista", accion=f"intento {intento}: FALLÓ", detalle=artefacto.error)

        return {
            "analysis_output": artefacto,
            "intentos_analista": intento,
            "messages": [AIMessage(content=resumen, name="analista")],
            "contribuciones": [{"agente": "analista", "intento": intento, "aporte": resumen}],
            "bitacora": [evento],
        }

    return nodo_analista
