"""
Supervisor: router inteligente y controlador de flujo.

Decide en tres capas, de la más determinística a la menos:

  1. Condición de parada dura: si se superó MAX_PASOS, corta (síntesis o escalado humano).
  2. Bucle de reparación: si la última validación falló, reenvía al MISMO especialista su
     salida y el error específico del validador; si ya agotó MAX_REINTENTOS, escala a humano.
  3. Decisión LLM con rúbrica de suficiencia (structured output → DecisionSupervisor),
     protegida por guardrails: no puede finalizar sin datos validados ni mandar al analista
     sin investigación aprobada.

Si el LLM del supervisor falla, usa una política determinística de respaldo, así el
"cuello de botella" de la topología jerárquica no paraliza todo el sistema.
Cada decisión queda en la bitácora con su razón y la versión del prompt.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from orquestador.agents.common import describir_error, formatear_hallazgo
from orquestador.agents.prompts import SUPERVISOR_PROMPT, SUPERVISOR_VERSION
from orquestador.state import (
    MAX_PASOS,
    MAX_REINTENTOS,
    AgentState,
    DecisionSupervisor,
    EventoBitacora,
    aprobado,
    ultima_validacion,
)


def resumen_del_estado(state: AgentState) -> str:
    """Contexto mínimo para el supervisor: artefactos y validaciones, no todo el historial."""
    lineas = [f"TAREA: {state['tarea']}", ""]
    research = state.get("research_output")
    if research and aprobado(state, "investigador"):
        lineas.append("HALLAZGOS VALIDADOS (investigador):")
        lineas += [
            f"  - {formatear_hallazgo(h)} [{h.fuente}]"
            for h in research.hallazgos
        ]
    else:
        lineas.append("HALLAZGOS VALIDADOS: ninguno todavía")
    analisis = state.get("analysis_output")
    if analisis and aprobado(state, "analista"):
        lineas.append(f"ANÁLISIS VALIDADO (analista): {analisis.expresion} = {analisis.resultado:g} — {analisis.explicacion}")
    else:
        lineas.append("ANÁLISIS VALIDADO: ninguno todavía")
    lineas.append(f"\nPaso {state.get('pasos', 0) + 1} de {MAX_PASOS}.")
    return "\n".join(lineas)


def prompt_de_reparacion(state: AgentState, agente: str, errores: list[str]) -> str:
    salida = state["research_output"] if agente == "investigador" else state["analysis_output"]
    salida_json = salida.model_dump_json(exclude={"evidencia"}, indent=2) if salida else "(sin salida)"
    return (
        "⚠️ TU RESPUESTA ANTERIOR NO PASÓ LA VALIDACIÓN AUTOMÁTICA.\n"
        f"Tu salida anterior fue:\n{salida_json}\n\n"
        "Errores específicos detectados:\n" + "\n".join(f"- {e}" for e in errores) + "\n\n"
        "Corregí SOLO esos puntos y devolvé la salida completa nuevamente."
    )


def politica_de_respaldo(state: AgentState) -> tuple[str, str]:
    """Ruteo determinístico si el LLM del supervisor no responde."""
    if not aprobado(state, "investigador"):
        return "investigador", "Buscá los datos de la empresa necesarios para responder la pregunta."
    if not aprobado(state, "analista"):
        return "analista", "Calculá lo que pide la pregunta usando los datos validados."
    return "sintesis", ""


def crear_nodo_supervisor(router):
    """`router` es un runnable con .ainvoke(messages) → DecisionSupervisor
    (ej: llm.with_structured_output(DecisionSupervisor))."""

    async def nodo_supervisor(state: AgentState) -> dict:
        pasos = state.get("pasos", 0) + 1

        def decidir(destino: str, razon: str, instruccion: str = "", feedback: str | None = None) -> dict:
            evento = EventoBitacora(
                nodo="supervisor",
                accion=f"paso {pasos}: → {destino}",
                detalle=f"{razon} [prompt {SUPERVISOR_VERSION}]" + (f" | instrucción: {instruccion}" if instruccion else ""),
            )
            return {
                "next_agent": destino,
                "instruccion": instruccion,
                "feedback_reparacion": feedback,
                "pasos": pasos,
                "task_completed": destino == "sintesis",
                "bitacora": [evento],
            }

        # 1) Condición de parada dura ------------------------------------------------
        if pasos > MAX_PASOS:
            hay_algo = aprobado(state, "investigador")
            return decidir(
                "sintesis" if hay_algo else "escalar_humano",
                f"Se alcanzó el máximo de {MAX_PASOS} pasos: se corta el ciclo "
                + ("y se sintetiza con lo validado." if hay_algo else "sin datos validados."),
            )

        # 2) Bucle de reparación ----------------------------------------------------
        ultima = ultima_validacion(state)
        if ultima and not ultima.aprobado:
            agente = ultima.agente
            intentos = state.get(f"intentos_{agente}", 0)
            if intentos > MAX_REINTENTOS:
                return decidir(
                    "escalar_humano",
                    f"{agente} falló la validación {intentos} veces (máx. {MAX_REINTENTOS} reparaciones): "
                    "error irresoluble, se escala a revisión humana.",
                )
            return decidir(
                agente,
                f"La salida de {agente} fue rechazada por el validador; se pide REFINAR "
                f"(reparación {intentos}/{MAX_REINTENTOS}).",
                instruccion=state.get("instruccion", ""),
                feedback=prompt_de_reparacion(state, agente, ultima.errores),
            )

        # 3) Decisión LLM con rúbrica ------------------------------------------------
        try:
            decision: DecisionSupervisor = await router.ainvoke(
                [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=resumen_del_estado(state))]
            )
        except Exception as e:
            destino, instruccion = politica_de_respaldo(state)
            return decidir(destino, f"LLM del supervisor falló ({describir_error(e)}); política de respaldo.", instruccion)

        destino = "sintesis" if decision.next == "FINISH" else decision.next
        razon = decision.razon

        # Guardrails sobre la decisión del LLM
        if destino == "sintesis" and not aprobado(state, "investigador"):
            destino, razon = "investigador", f"OVERRIDE: se pidió FINISH sin datos validados (R1). LLM dijo: {razon}"
        elif destino == "analista" and not aprobado(state, "investigador"):
            destino, razon = "investigador", f"OVERRIDE: el analista necesita datos validados primero. LLM dijo: {razon}"

        instruccion = decision.instruccion or politica_de_respaldo(state)[1]
        return decidir(destino, razon, instruccion if destino != "sintesis" else "")

    return nodo_supervisor
