"""Utilidades compartidas por los agentes."""

from __future__ import annotations

import os

from langchain_core.messages import BaseMessage


def crear_llm(temperature: float = 0.0):
    """LLM por defecto del proyecto (Gemini). El modelo se puede cambiar con GEMINI_MODEL."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    extra = {}
    if os.getenv("GEMINI_RPM"):
        # Rate limiter del lado del cliente: con N tareas concurrentes, las llamadas esperan su turno
        # en vez de chocar contra el límite por minuto de la API (429). El grafo crea UNA instancia del
        # LLM que comparten todos los agentes, así el límite vale para todo el proceso del worker.
        from langchain_core.rate_limiters import InMemoryRateLimiter

        extra["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=float(os.environ["GEMINI_RPM"]) / 60, check_every_n_seconds=0.1, max_bucket_size=1
        )
    if os.getenv("GEMINI_THINKING_LEVEL"):  # opcional: "minimal" / "low" bajan la latencia de Gemini 3+
        extra["thinking_level"] = os.environ["GEMINI_THINKING_LEVEL"]
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"), temperature=temperature, **extra
    )


def texto_de(mensaje: BaseMessage | str) -> str:
    """Normaliza el contenido de un mensaje a texto.

    Gemini (vía langchain-google-genai) puede devolver `content` como una lista de bloques
    [{'type': 'text', 'text': '...', 'extras': {'signature': ...}}]. La firma es metadata
    interna del modelo, no parte de la respuesta: nos quedamos solo con el texto.
    """
    contenido = mensaje if isinstance(mensaje, str) else mensaje.content
    if isinstance(contenido, str):
        return contenido
    partes = []
    for bloque in contenido:
        if isinstance(bloque, str):
            partes.append(bloque)
        elif isinstance(bloque, dict) and bloque.get("type") == "text":
            partes.append(bloque.get("text", ""))
    return "".join(partes)


def formatear_hallazgo(h) -> str:
    """'Vacaciones con 5 a 10 años: 21 días' (sin el '.0' de los floats)."""
    valor = f"{h.valor:g}" if h.valor is not None else "(cualitativo)"
    return f"{h.dato}: {valor} {h.unidad or ''}".rstrip()


def describir_error(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:300]}"
