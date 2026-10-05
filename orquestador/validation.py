"""
Puerta de calidad determinística ("Pydantic, no vibras").

Se ejecuta después de CADA especialista. No usa un LLM: verifica reglas con código, así
un texto que "suena profesional" pero está mal no pasa. Los errores que produce son
ESPECÍFICOS, porque el supervisor se los reenvía al agente en el bucle de reparación.

Resolución de conflictos por PROCEDENCIA (ni promedio ni mayoría):
  - Un hallazgo del Investigador vale solo si su fuente y su número aparecen en la
    evidencia que devolvió realmente la herramienta de búsqueda.
  - Un número usado por el Analista vale solo si proviene de un hallazgo validado o de
    la pregunta del usuario. Si el Analista "contradice" al Investigador usando otro
    número, el validador lo rechaza porque ese número no tiene procedencia.
"""

from __future__ import annotations

import math
import re

from orquestador.state import (
    UMBRAL_CONFIANZA,
    AgentState,
    AnalysisArtifact,
    EventoBitacora,
    ResearchArtifact,
    ValidationReport,
    aprobado,
)
from orquestador.tools.calculator import evaluar_expresion, numeros_en

# Constantes "estructurales" que el analista puede usar sin que vengan de un documento
# (ej: 2 años restantes, dividir por 100 un porcentaje, 12 meses).
CONSTANTES_PERMITIDAS = {float(n) for n in range(0, 13)} | {100.0}

_PATRON_NUMERO = re.compile(r"\d+(?:[.,]\d+)?")


def numeros_del_texto(texto: str) -> set[float]:
    return {float(n.replace(",", ".")) for n in _PATRON_NUMERO.findall(texto or "")}


def _contiene(conjunto: set[float], valor: float) -> bool:
    return any(math.isclose(valor, x, rel_tol=1e-9, abs_tol=1e-9) for x in conjunto)


def validar_investigacion(art: ResearchArtifact | None) -> list[str]:
    if art is None:
        return ["El investigador no produjo ningún resultado."]
    if art.error:
        return [f"El investigador falló con una excepción: {art.error}"]

    errores = []
    if not art.fuentes_recuperadas:
        errores.append("No usaste la herramienta `buscar_en_politicas_internas`: todo dato debe salir de una búsqueda.")
    if not art.hallazgos:
        errores.append("No reportaste ningún hallazgo.")
    if art.confidence_score < UMBRAL_CONFIANZA:
        errores.append(
            f"confidence_score={art.confidence_score:.2f} está por debajo del umbral {UMBRAL_CONFIANZA}: "
            "reformulá la búsqueda para encontrar el dato faltante."
        )

    numeros_evidencia = numeros_del_texto(art.evidencia)
    for h in art.hallazgos:
        if art.fuentes_recuperadas and h.fuente not in art.fuentes_recuperadas:
            errores.append(
                f"El hallazgo '{h.dato}' cita la fuente '{h.fuente}', que la búsqueda NO devolvió. "
                f"Fuentes realmente recuperadas: {art.fuentes_recuperadas}."
            )
        if h.valor is not None and art.evidencia and not _contiene(numeros_evidencia, h.valor):
            errores.append(
                f"El valor {h.valor:g} del hallazgo '{h.dato}' no aparece en el texto recuperado: posible alucinación."
            )
    return errores


def validar_analisis(art: AnalysisArtifact | None, state: AgentState) -> list[str]:
    if art is None:
        return ["El analista no produjo ningún resultado."]
    if art.error:
        return [f"El analista falló con una excepción: {art.error}"]
    if not aprobado(state, "investigador"):
        return ["El análisis se hizo sin datos validados del investigador."]

    errores = []
    try:
        recalculado = evaluar_expresion(art.expresion)
    except Exception as e:
        return [f"La expresión '{art.expresion}' no es aritmética válida: {e}"]

    if not math.isclose(recalculado, art.resultado, rel_tol=1e-6, abs_tol=1e-6):
        errores.append(
            f"La expresión '{art.expresion}' da {recalculado:g}, pero declaraste resultado={art.resultado:g}. "
            "Volvé a ejecutar la calculadora y copiá su resultado."
        )

    research = state["research_output"]
    permitidos = (
        CONSTANTES_PERMITIDAS
        | {float(h.valor) for h in research.hallazgos if h.valor is not None}
        | numeros_del_texto(state["tarea"])
    )
    sin_procedencia = sorted({n for n in numeros_en(art.expresion) if not _contiene(permitidos, n)})
    if sin_procedencia:
        errores.append(
            f"Los números {[f'{n:g}' for n in sin_procedencia]} de la expresión no provienen de ningún hallazgo "
            "validado ni de la pregunta. Usá solo los DATOS VALIDADOS."
        )

    if art.confidence_score < UMBRAL_CONFIANZA:
        errores.append(f"confidence_score={art.confidence_score:.2f} está por debajo del umbral {UMBRAL_CONFIANZA}.")
    return errores


async def nodo_validador(state: AgentState) -> dict:
    agente = state["next_agent"]  # el especialista que acaba de ejecutarse
    if agente == "investigador":
        errores = validar_investigacion(state.get("research_output"))
    else:
        errores = validar_analisis(state.get("analysis_output"), state)

    reporte = ValidationReport(agente=agente, aprobado=not errores, errores=errores)
    evento = EventoBitacora(
        nodo="validador",
        accion=f"{agente}: {'APROBADO' if reporte.aprobado else 'RECHAZADO'}",
        detalle=" | ".join(errores),
    )
    return {"validaciones": [reporte], "bitacora": [evento]}
