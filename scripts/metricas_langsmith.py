"""
Lectura del dashboard por API: costo por ejecución, latencia p50/p95 y tokens por nodo de una corrida.

    python scripts/metricas_langsmith.py --corrida <id>     # id que imprime carga_concurrente.py
    python scripts/metricas_langsmith.py --ultimas 10       # últimas N trazas raíz del proyecto

Por qué existe: los dashboards de LangSmith muestran latencia P50 y P99, no P95. Este script toma
los MISMOS runs que muestra el dashboard (con el costo que LangSmith calculó a partir de los tokens
de entrada/salida) y calcula p95. Sus números deben coincidir con los de la UI.

Nota sobre HITL: un job crítico genera DOS trazas raíz (ejecución + reanudación) unidas por el
mismo job_id/thread_id. El costo "por ejecución" se reporta por job (suma de sus trazas) y la
latencia por traza (tiempo de cómputo del agente, sin contar la espera del humano).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

from dotenv import load_dotenv
from langsmith import Client

NODOS = ["supervisor", "investigador", "analista", "validador", "aprobacion_humana", "sintesis", "rechazado",
         "escalar_humano"]


def percentil(valores: list[float], p: float) -> float:
    orden = sorted(valores)
    if not orden:
        return float("nan")
    k = (len(orden) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(orden) - 1)
    return orden[f] + (orden[c] - orden[f]) * (k - f)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    parser = argparse.ArgumentParser()
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--corrida", help="id de la corrida de carga (tag carga:<id>)")
    grupo.add_argument("--ultimas", type=int, help="analizar las últimas N trazas raíz")
    parser.add_argument("--proyecto", default=os.getenv("LANGSMITH_PROJECT", "api-agente-produccion"))
    args = parser.parse_args()

    client = Client()
    filtro = f'and(eq(name, "tarea_agente"), has(tags, "carga:{args.corrida}"))' if args.corrida else 'eq(name, "tarea_agente")'
    raices = list(client.list_runs(project_name=args.proyecto, is_root=True, filter=filtro, limit=args.ultimas))
    if not raices:
        print("No se encontraron trazas (¿las trazas tardan unos segundos en aparecer? ¿proyecto correcto?)")
        return

    # --- Por traza --------------------------------------------------------------------------
    print(f"Proyecto: {args.proyecto} · {len(raices)} trazas raíz\n")
    print(f"{'job_id':<34}{'fase':<13}{'status':<9}{'latencia':>10}{'tokens':>9}{'costo USD':>12}")
    por_job: dict[str, dict] = defaultdict(lambda: {"costo": 0.0, "tokens": 0, "latencia": 0.0})
    latencias, sin_costo = [], 0
    for r in sorted(raices, key=lambda r: r.start_time):
        meta = (r.extra or {}).get("metadata", {})
        job, fase = meta.get("job_id", "?"), meta.get("fase", "?")
        lat = (r.end_time - r.start_time).total_seconds() if r.end_time else float("nan")
        costo = float(r.total_cost) if r.total_cost is not None else 0.0
        sin_costo += r.total_cost is None
        latencias.append(lat)
        por_job[job]["costo"] += costo
        por_job[job]["tokens"] += r.total_tokens or 0
        por_job[job]["latencia"] += lat
        print(f"{job:<34}{fase:<13}{r.status:<9}{lat:>9.2f}s{r.total_tokens or 0:>9}{costo:>12.6f}")

    costos_job = [j["costo"] for j in por_job.values()]
    print("\n" + "=" * 87)
    print(f"Latencia por traza      p50={percentil(latencias, 50):.2f}s   p95={percentil(latencias, 95):.2f}s   "
          f"p99={percentil(latencias, 99):.2f}s")
    print(f"Costo por ejecución     promedio={sum(costos_job) / len(costos_job):.6f} USD   "
          f"máx={max(costos_job):.6f} USD   total corrida={sum(costos_job):.6f} USD  ({len(por_job)} jobs)")
    if sin_costo:
        print(f"⚠️  {sin_costo} trazas sin costo: LangSmith no tiene precio para el modelo. Cargalo en "
              "Settings → Model Pricing (ver README).")

    # --- Por nodo: hijos directos del grafo -------------------------------------------------
    tokens_nodo, lat_nodo, llamadas = defaultdict(int), defaultdict(list), defaultdict(int)
    for raiz in raices:
        runs = list(client.list_runs(project_name=args.proyecto, trace_id=raiz.trace_id))
        grafos = {r.id for r in runs if r.name == "orquestador_multiagente"}
        for r in runs:
            if r.parent_run_id in grafos and r.name in NODOS:
                tokens_nodo[r.name] += r.total_tokens or 0
                llamadas[r.name] += 1
                if r.end_time:
                    lat_nodo[r.name].append((r.end_time - r.start_time).total_seconds())

    total_tokens = sum(tokens_nodo.values()) or 1
    print(f"\n{'nodo':<20}{'ejecuciones':>12}{'tokens':>10}{'% tokens':>10}{'lat. media':>12}{'lat. total':>12}")
    for nodo in sorted(tokens_nodo, key=lambda n: -tokens_nodo[n]):
        lats = lat_nodo[nodo]
        print(f"{nodo:<20}{llamadas[nodo]:>12}{tokens_nodo[nodo]:>10}{100 * tokens_nodo[nodo] / total_tokens:>9.1f}%"
              f"{(sum(lats) / len(lats) if lats else 0):>11.2f}s{sum(lats):>11.2f}s")


if __name__ == "__main__":
    main()
