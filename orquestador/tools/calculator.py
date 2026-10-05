"""
Calculadora segura: parsea la expresión con `ast` y solo permite aritmética.

Usar eval() sobre texto generado por un LLM es ejecución de código arbitrario.
Acá no hay imports, llamadas a funciones, atributos ni builtins: solo números y
+ - * / ** % y paréntesis. Se reutiliza también en el validador para RECALCULAR
de forma determinística el resultado que declara el Analista.
"""

from __future__ import annotations

import ast
import operator as op

_BINARIOS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}
_UNARIOS = {ast.USub: op.neg, ast.UAdd: op.pos}


def _evaluar(nodo: ast.AST) -> float:
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)) and not isinstance(nodo.value, bool):
        return nodo.value
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _BINARIOS:
        izq, der = _evaluar(nodo.left), _evaluar(nodo.right)
        if isinstance(nodo.op, ast.Pow) and abs(der) > 100:
            raise ValueError("exponente demasiado grande")
        return _BINARIOS[type(nodo.op)](izq, der)
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _UNARIOS:
        return _UNARIOS[type(nodo.op)](_evaluar(nodo.operand))
    raise ValueError(f"elemento no permitido: {type(nodo).__name__}")


def evaluar_expresion(expresion: str) -> float:
    """Evalúa una expresión aritmética o lanza ValueError."""
    return _evaluar(ast.parse(expresion.strip(), mode="eval").body)


def numeros_en(expresion: str) -> list[float]:
    """Literales numéricos que aparecen en una expresión (para chequear procedencia)."""
    return [
        float(n.value)
        for n in ast.walk(ast.parse(expresion.strip(), mode="eval"))
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
    ]
