"""Utilidades compartidas por los agentes."""

from __future__ import annotations

import os

from langchain_core.messages import BaseMessage


def crear_llm(temperature: float = 0.0):
    """LLM por defecto del proyecto (Gemini). El modelo se puede cambiar con GEMINI_MODEL."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"), temperature=temperature)


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
