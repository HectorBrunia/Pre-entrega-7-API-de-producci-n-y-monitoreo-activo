"""
Agente de Búsqueda/Investigación.

- Herramienta acotada: solo puede consultar el RAG de políticas internas.
- Salida estructurada: devuelve un `ResearchFindings` (Pydantic).
- Contexto acotado: recibe SOLO la instrucción del supervisor (+ feedback de reparación
  si su intento anterior falló la validación), nunca todo el historial.
- Escribe SOLO en su parcela: `research_output` e `intentos_investigador`.
"""

from __future__ import annotations

import re

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from orquestador.agents.common import describir_error, formatear_hallazgo, texto_de
from orquestador.agents.prompts import INVESTIGADOR_PROMPT
from orquestador.state import AgentState, EventoBitacora, ResearchArtifact, ResearchFindings

_PATRON_FUENTE = re.compile(r"\[Fuente: ([^\]]+)\]")


@tool
def buscar_en_politicas_internas(consulta: str) -> str:
    """Busca en las políticas internas de TechCorp (vacaciones, teletrabajo, seguridad
    informática, onboarding, capacitación, horas extra). Devuelve fragmentos con su fuente."""
    from orquestador.tools.knowledge_base import buscar

    resultados = buscar(consulta)
    if not resultados:
        return "No se encontró información relevante en las políticas."
    return "\n\n".join(f"[Fuente: {r['fuente']}] {r['contenido']}" for r in resultados)


def crear_agente_investigador(llm):
    return create_agent(
        model=llm,
        tools=[buscar_en_politicas_internas],
        system_prompt=INVESTIGADOR_PROMPT,
        response_format=ToolStrategy(ResearchFindings),
        name="investigador",
    )


def extraer_evidencia(mensajes) -> tuple[list[str], str]:
    """Fuentes y texto EFECTIVAMENTE devueltos por la herramienta (no lo que dice el LLM)."""
    salidas = [texto_de(m) for m in mensajes if isinstance(m, ToolMessage) and m.name == buscar_en_politicas_internas.name]
    evidencia = "\n\n".join(salidas)
    fuentes = sorted(set(_PATRON_FUENTE.findall(evidencia)))
    return fuentes, evidencia


def armar_tarea(state: AgentState) -> str:
    tarea = state["instruccion"] or state["tarea"]
    if state.get("feedback_reparacion"):
        tarea += "\n\n" + state["feedback_reparacion"]
    return tarea


def crear_nodo_investigador(agente):
    """`agente` es cualquier runnable con .ainvoke({'messages': [...]}) → {'messages', 'structured_response'}."""

    async def nodo_investigador(state: AgentState) -> dict:
        intento = state.get("intentos_investigador", 0) + 1
        try:
            resultado = await agente.ainvoke({"messages": [HumanMessage(content=armar_tarea(state))]})
            fuentes, evidencia = extraer_evidencia(resultado["messages"])
            findings: ResearchFindings = resultado["structured_response"]
            artefacto = ResearchArtifact(**findings.model_dump(), fuentes_recuperadas=fuentes, evidencia=evidencia)
            resumen = artefacto.resumen + " | " + "; ".join(
                f"{formatear_hallazgo(h)} ({h.fuente})"
                for h in artefacto.hallazgos
            )
            evento = EventoBitacora(
                nodo="investigador",
                accion=f"intento {intento}: {len(artefacto.hallazgos)} hallazgo(s)",
                detalle=f"confianza={artefacto.confidence_score:.2f}; fuentes recuperadas={fuentes}",
            )
        except Exception as e:  # un nodo que explota no debe dejar el grafo colgado
            artefacto = ResearchArtifact.model_construct(
                hallazgos=[], resumen="", confidence_score=0.0, fuentes_recuperadas=[], evidencia="",
                error=describir_error(e),
            )
            resumen = f"ERROR: {artefacto.error}"
            evento = EventoBitacora(nodo="investigador", accion=f"intento {intento}: FALLÓ", detalle=artefacto.error)

        return {
            "research_output": artefacto,
            "intentos_investigador": intento,
            "messages": [AIMessage(content=resumen, name="investigador")],
            "contribuciones": [{"agente": "investigador", "intento": intento, "aporte": resumen}],
            "bitacora": [evento],
        }

    return nodo_investigador
