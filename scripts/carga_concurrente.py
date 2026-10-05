"""
Prueba de carga: 5 peticiones concurrentes contra la API.

    python scripts/carga_concurrente.py                       # aprueba automáticamente las tareas críticas
    python scripts/carga_concurrente.py --sin-auto-aprobar    # deja las críticas esperando (aprobalas a mano)
    python scripts/carga_concurrente.py --url http://localhost:8000 -n 5

Cada petición lleva la etiqueta `carga:<id-de-corrida>`, así en LangSmith se filtran exactamente
las trazas de esta corrida (y `scripts/metricas_langsmith.py --corrida <id>` calcula costo y p95).
De las 5 preguntas, 3 involucran montos en USD → pasan por el nodo de aprobación humana.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx

PREGUNTAS = [
    # no crítica
    "Un empleado va a cumplir 12 años en TechCorp. Tuvo los días de vacaciones que le correspondían según "
    "cada tramo de antigüedad: los primeros 5 años, los siguientes 5 y los últimos 2. ¿Cuántos días de "
    "vacaciones acumuló en total en esos 12 años?",
    # crítica (USD)
    "Un empleado cuya hora normal vale 12 USD hizo este mes 10 horas extra en días hábiles y 6 horas extra en "
    "un feriado. Según la política de TechCorp, ¿cuánto cobra por esas horas extra?",
    # crítica (USD)
    "Un líder de equipo de TechCorp no usó su presupuesto de capacitación el año pasado. Este año quiere hacer "
    "un curso de 2300 USD. Considerando lo máximo que puede acumular, ¿cuánto dinero le sobra o le falta?",
    # no crítica
    "Un empleado de TechCorp en esquema híbrido estándar, ¿cuántos días presenciales hace en 4 semanas?",
    # crítica (USD)
    "¿Cuánto dinero recibe en total en un año un empleado híbrido de TechCorp por el reintegro de conectividad?",
]

TERMINALES = {"DONE", "FAILED"}


def percentil(valores: list[float], p: float) -> float:
    """Percentil con interpolación lineal (mismo criterio que numpy por defecto)."""
    orden = sorted(valores)
    if len(orden) == 1:
        return orden[0]
    k = (len(orden) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(orden) - 1)
    return orden[f] + (orden[c] - orden[f]) * (k - f)


async def una_peticion(cliente: httpx.AsyncClient, i: int, pregunta: str, corrida: str, auto_aprobar: bool) -> dict:
    t0 = time.perf_counter()
    resp = await cliente.post("/tasks", json={"pregunta": pregunta, "etiquetas": [f"carga:{corrida}"]})
    resp.raise_for_status()
    t_encolado = time.perf_counter() - t0
    job_id = resp.json()["job_id"]
    print(f"[{i}] encolado en {t_encolado * 1000:.0f} ms → {job_id}")

    espera_humana = 0.0
    estados_vistos = []
    while True:
        job = (await cliente.get(f"/tasks/{job_id}")).json()
        if not estados_vistos or estados_vistos[-1] != job["status"]:
            estados_vistos.append(job["status"])
            print(f"[{i}] {job['status']}")
        if job["status"] in TERMINALES:
            break
        if job["status"] == "WAITING_APPROVAL" and auto_aprobar:
            t_h = time.perf_counter()
            await cliente.post(
                f"/tasks/{job_id}/approve",
                json={"aprobado": True, "revisor": "carga-concurrente", "comentario": "aprobación automática del test"},
            )
            espera_humana += time.perf_counter() - t_h
        await asyncio.sleep(1.0)

    total = time.perf_counter() - t0
    return {
        "n": i,
        "job_id": job_id,
        "status": job["status"],
        "estados": estados_vistos,
        "encolado_ms": round(t_encolado * 1000, 1),
        "latencia_total_s": round(total, 2),
        "paso_por_hitl": "WAITING_APPROVAL" in estados_vistos,
        "error": job.get("error"),
        "respuesta": (job.get("resultado") or {}).get("respuesta_final"),
    }


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("-n", type=int, default=5)
    parser.add_argument("--sin-auto-aprobar", action="store_true")
    args = parser.parse_args()

    corrida = uuid.uuid4().hex[:8]
    preguntas = [PREGUNTAS[i % len(PREGUNTAS)] for i in range(args.n)]
    print(f"Corrida {corrida}: {args.n} peticiones concurrentes contra {args.url}\n")

    async with httpx.AsyncClient(base_url=args.url, timeout=30) as cliente:
        t0 = time.perf_counter()
        resultados = await asyncio.gather(
            *(una_peticion(cliente, i + 1, p, corrida, not args.sin_auto_aprobar) for i, p in enumerate(preguntas))
        )
        pared = time.perf_counter() - t0

    latencias = [r["latencia_total_s"] for r in resultados]
    encolado = [r["encolado_ms"] for r in resultados]
    print("\n" + "=" * 78)
    print(f"{'#':<3}{'status':<10}{'HITL':<6}{'encolado':>10}{'total (s)':>11}  job_id")
    for r in resultados:
        print(f"{r['n']:<3}{r['status']:<10}{'sí' if r['paso_por_hitl'] else 'no':<6}"
              f"{r['encolado_ms']:>8.0f}ms{r['latencia_total_s']:>11.2f}  {r['job_id']}")
    print("=" * 78)
    print(f"POST /tasks (encolar)  p50={statistics.median(encolado):.0f} ms  máx={max(encolado):.0f} ms")
    print(f"Latencia end-to-end    p50={percentil(latencias, 50):.2f} s  p95={percentil(latencias, 95):.2f} s")
    print(f"Tiempo de pared total  {pared:.2f} s (si fuera secuencial: ~{sum(latencias):.0f} s)")
    print(f"\nFiltro en LangSmith:   tag = carga:{corrida}")
    print(f"Métricas de la corrida: python scripts/metricas_langsmith.py --corrida {corrida}")

    salida = Path("resultados") / f"carga_{corrida}.json"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(json.dumps({"corrida": corrida, "resultados": resultados}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detalle guardado en {salida}")


if __name__ == "__main__":
    asyncio.run(main())
