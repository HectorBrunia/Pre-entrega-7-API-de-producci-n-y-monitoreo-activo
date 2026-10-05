"""
Nodos terminales: síntesis final y escalado a revisión humana.

La síntesis solo usa artefactos APROBADOS por el validador y agrega al final, de forma
determinística, el resultado verificado y las fuentes (trazabilidad para el usuario).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from orquestador.agents.common import describir_error, formatear_hallazgo, texto_de
from orquestador.agents.prompts import SINTESIS_PROMPT
from orquestador.state import AgentState, EventoBitacora, aprobado, ultima_validacion


def _aportes_validados(state: AgentState) -> tuple[str, list[str]]:
    partes, fuentes = [], []
    research = state.get("research_output")
    if research and aprobado(state, "investigador"):
        for h in research.hallazgos:
            partes.append(f"- [investigador] {formatear_hallazgo(h)} ({h.fuente})")
            fuentes.append(h.fuente)
    analisis = state.get("analysis_output")
    if analisis and aprobado(state, "analista"):
        partes.append(f"- [analista] {analisis.expresion} = {analisis.resultado:g}. {analisis.explicacion}")
    return "\n".join(partes), sorted(set(fuentes))


def crear_nodo_sintesis(redactor):
    async def nodo_sintesis(state: AgentState) -> dict:
        aportes, fuentes = _aportes_validados(state)
        try:
            respuesta = await redactor.ainvoke(
                [
                    SystemMessage(content=SINTESIS_PROMPT),
                    HumanMessage(content=f"Pregunta: {state['tarea']}\n\nAportes validados:\n{aportes}"),
                ]
            )
            texto = texto_de(respuesta).strip()
            accion = "respuesta final redactada"
        except Exception as e:
            texto = f"(No se pudo redactar con el LLM: {describir_error(e)}). Datos validados:\n{aportes}"
            accion = "redacción LLM falló; se entregan los datos validados en crudo"

        analisis = state.get("analysis_output")
        pie = []
        if analisis and aprobado(state, "analista"):
            pie.append(f"Cálculo verificado: {analisis.expresion} = {analisis.resultado:g}")
        if fuentes:
            pie.append("Fuentes: " + ", ".join(fuentes))
        aprobacion = state.get("aprobacion")
        if aprobacion and aprobacion.get("aprobado"):
            nota = f": {aprobacion['comentario']}" if aprobacion.get("comentario") else ""
            pie.append(f"Aprobado por revisor humano ({aprobacion.get('revisor') or 'anónimo'}){nota}")
        final = texto + ("\n\n" + "\n".join(pie) if pie else "")

        return {
            "respuesta_final": final,
            "task_completed": True,
            "messages": [AIMessage(content=final, name="sintesis")],
            "bitacora": [EventoBitacora(nodo="sintesis", accion=accion, detalle=f"fuentes={fuentes}")],
        }

    return nodo_sintesis


async def nodo_escalar_humano(state: AgentState) -> dict:
    """Human-in-the-loop: se detiene y deja un ticket con toda la evidencia del fallo.

    En producción se usaría `langgraph.types.interrupt()` con un checkpointer para pausar
    el grafo hasta que una persona apruebe/corrija; acá se genera el ticket y se termina.
    """
    ultima = ultima_validacion(state)
    motivo = (
        f"El agente '{ultima.agente}' no superó la validación: " + "; ".join(ultima.errores)
        if ultima and not ultima.aprobado
        else "Se alcanzó el límite de pasos sin datos suficientes."
    )
    texto = (
        "🧑‍💼 ESCALADO A REVISIÓN HUMANA\n"
        f"Pregunta: {state['tarea']}\n"
        f"Motivo: {motivo}\n"
        f"Pasos ejecutados: {state.get('pasos', 0)} | intentos investigador: {state.get('intentos_investigador', 0)} "
        f"| intentos analista: {state.get('intentos_analista', 0)}\n"
        "No se entrega una respuesta automática para evitar informar un dato no verificado."
    )
    return {
        "respuesta_final": texto,
        "escalado_humano": True,
        "task_completed": False,
        "messages": [AIMessage(content=texto, name="escalar_humano")],
        "bitacora": [EventoBitacora(nodo="escalar_humano", accion="ticket generado", detalle=motivo)],
    }
