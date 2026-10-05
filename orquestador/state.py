"""
state.py — Estado compartido del orquestador y contratos de datos entre agentes.

Tres ideas de la clase aplicadas acá:

1. Estado como CONTRATO: lo que un agente le pasa a otro no es texto libre sino un
   modelo Pydantic (ResearchArtifact, AnalysisArtifact). Si el LLM alucina un formato
   o un score fuera de rango, el error se captura al construir el artefacto.

2. AISLAMIENTO DE ESCRITURA: cada nodo es dueño de una "parcela" del estado
   (ver PERMISOS_ESCRITURA). `verificar_escritura()` se aplica a cada update en
   graph.py, así ningún agente puede pisar lo que calculó otro.

3. TRAZABILIDAD: `contribuciones` y `bitacora` usan `operator.add` como reducer,
   así cada nodo SUMA su aporte/decisión en lugar de sobrescribir. Al final queda el
   log completo de quién decidió qué y por qué.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Nombres de nodos (única fuente de verdad para el ruteo)
# ---------------------------------------------------------------------------
Especialista = Literal["investigador", "analista"]
DestinoSupervisor = Literal["investigador", "analista", "sintesis", "escalar_humano"]
DestinoRuteo = Literal["investigador", "analista", "sintesis", "escalar_humano", "aprobacion_humana"]

MAX_PASOS = 8          # corte duro del "Supervisor Infinito"
MAX_REINTENTOS = 2     # reparaciones por especialista antes de escalar a humano
UMBRAL_CONFIANZA = 0.5  # confidence_score mínimo aceptado por el validador


# ---------------------------------------------------------------------------
# Contratos del Investigador
# ---------------------------------------------------------------------------
class Hallazgo(BaseModel):
    """Un dato concreto encontrado en las políticas, con su procedencia."""

    dato: str = Field(description="Descripción corta del dato, ej: 'Vacaciones con 5 a 10 años de antigüedad'")
    valor: Optional[float] = Field(
        default=None, description="Valor numérico del dato si lo tiene (ej: 21). Null si es cualitativo."
    )
    unidad: Optional[str] = Field(default=None, description="Unidad del valor, ej: 'días', 'USD', 'horas'")
    fuente: str = Field(description="Nombre EXACTO del archivo de donde salió el dato, ej: 'politica_vacaciones.txt'")


class ResearchFindings(BaseModel):
    """Lo que el LLM investigador debe devolver (schema de salida estructurada)."""

    hallazgos: list[Hallazgo] = Field(min_length=1, description="Datos encontrados, uno por hallazgo")
    resumen: str = Field(description="Resumen breve de lo encontrado")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confianza entre 0 y 1 en que los datos responden la tarea")


class ResearchArtifact(ResearchFindings):
    """ResearchFindings + evidencia real recuperada por la herramienta.

    `fuentes_recuperadas` y `evidencia` NO los escribe el LLM: los completa el nodo
    a partir de los ToolMessage, y el validador los usa para chequear procedencia.
    """

    fuentes_recuperadas: list[str] = Field(default_factory=list)
    evidencia: str = ""
    error: Optional[str] = None  # excepción capturada, si la hubo


# ---------------------------------------------------------------------------
# Contratos del Analista
# ---------------------------------------------------------------------------
class AnalysisResult(BaseModel):
    """Lo que el LLM analista debe devolver (schema de salida estructurada)."""

    expresion: str = Field(description="Expresión aritmética EXACTA usada con la calculadora, ej: '5*14 + 5*21 + 2*28'")
    resultado: float = Field(description="Resultado numérico devuelto por la calculadora")
    explicacion: str = Field(description="Qué representa el cálculo y de dónde sale cada número")
    confidence_score: float = Field(ge=0.0, le=1.0)

    @field_validator("expresion")
    @classmethod
    def solo_aritmetica(cls, v: str) -> str:
        permitidos = set("0123456789.+-*/() ")
        invalidos = set(v) - permitidos
        if invalidos:
            raise ValueError(f"la expresión contiene caracteres no aritméticos: {sorted(invalidos)}")
        return v


class AnalysisArtifact(AnalysisResult):
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Contratos del Validador y del Supervisor
# ---------------------------------------------------------------------------
class ValidationReport(BaseModel):
    agente: Especialista
    aprobado: bool
    errores: list[str] = Field(default_factory=list)


class DecisionSupervisor(BaseModel):
    """Salida estructurada del Supervisor: ¿quién interviene ahora o se termina?"""

    next: Literal["investigador", "analista", "FINISH"] = Field(
        description="Próximo especialista a invocar, o FINISH si la tarea ya está resuelta y validada"
    )
    razon: str = Field(description="Justificación breve de la decisión, citando la rúbrica")
    instruccion: str = Field(
        default="",
        description="Instrucción puntual y autocontenida para el especialista elegido (vacío si FINISH)",
    )


class EventoBitacora(BaseModel):
    """Una entrada del log de auditoría: quién decidió qué y por qué."""

    nodo: str
    accion: str
    detalle: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


# ---------------------------------------------------------------------------
# Estado del grafo
# ---------------------------------------------------------------------------
class AgentState(MessagesState):
    """Hereda `messages` (reducer add_messages) y suma campos propios."""

    tarea: str                                   # pregunta original del usuario (inmutable)

    # --- parcela del Supervisor ---
    next_agent: Optional[DestinoSupervisor]
    instruccion: str                             # instrucción acotada para el especialista
    feedback_reparacion: Optional[str]           # error específico a corregir (bucle de reparación)
    pasos: int
    task_completed: bool

    # --- parcela del Investigador ---
    research_output: Optional[ResearchArtifact]
    intentos_investigador: int

    # --- parcela del Analista ---
    analysis_output: Optional[AnalysisArtifact]
    intentos_analista: int

    # --- parcela del Validador ---
    validaciones: Annotated[list[ValidationReport], operator.add]

    # --- parcela de la Aprobación Humana (HITL) ---
    aprobacion: Optional[dict]                   # {"aprobado", "revisor", "comentario", "motivos"}

    # --- parcela de Síntesis / Escalado ---
    respuesta_final: Optional[str]
    escalado_humano: bool

    # --- compartidos append-only (todos suman, nadie pisa) ---
    contribuciones: Annotated[list[dict], operator.add]
    bitacora: Annotated[list[EventoBitacora], operator.add]


COMPARTIDOS = {"messages", "contribuciones", "bitacora"}

PERMISOS_ESCRITURA: dict[str, set[str]] = {
    "supervisor": {"next_agent", "instruccion", "feedback_reparacion", "pasos", "task_completed"},
    "investigador": {"research_output", "intentos_investigador"},
    "analista": {"analysis_output", "intentos_analista"},
    "validador": {"validaciones"},
    "sintesis": {"respuesta_final", "task_completed"},
    "escalar_humano": {"respuesta_final", "escalado_humano", "task_completed"},
    "aprobacion_humana": {"aprobacion"},
    "rechazado": {"respuesta_final", "task_completed"},
}


class ViolacionDeEscritura(RuntimeError):
    """Un nodo intentó escribir fuera de su parcela del estado."""


def verificar_escritura(nodo: str, update: dict) -> dict:
    """Aislamiento de escritura: falla en forma ruidosa si un nodo pisa campos ajenos."""
    permitidos = PERMISOS_ESCRITURA[nodo] | COMPARTIDOS
    ajenos = set(update) - permitidos
    if ajenos:
        raise ViolacionDeEscritura(f"El nodo '{nodo}' intentó escribir campos ajenos: {sorted(ajenos)}")
    return update


def estado_inicial(pregunta: str) -> dict:
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=pregunta)],
        "tarea": pregunta,
        "next_agent": None,
        "instruccion": "",
        "feedback_reparacion": None,
        "pasos": 0,
        "task_completed": False,
        "research_output": None,
        "intentos_investigador": 0,
        "analysis_output": None,
        "intentos_analista": 0,
        "validaciones": [],
        "respuesta_final": None,
        "escalado_humano": False,
        "aprobacion": None,
        "contribuciones": [],
        "bitacora": [],
    }


def ultima_validacion(state: dict, agente: Optional[str] = None) -> Optional[ValidationReport]:
    for v in reversed(state.get("validaciones") or []):
        if agente is None or v.agente == agente:
            return v
    return None


def aprobado(state: dict, agente: str) -> bool:
    v = ultima_validacion(state, agente)
    return bool(v and v.aprobado)
