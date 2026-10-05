"""Configuración leída del entorno (.env). Un solo lugar para todos los parámetros operativos."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL") or "redis://localhost:6379/0"  # "" (vacío en .env) también cae al default

# Cola de Arq (API y worker deben usar la misma)
COLA = os.getenv("ARQ_QUEUE", "arq:agente")
MAX_JOBS_CONCURRENTES = int(os.getenv("WORKER_MAX_JOBS", "10"))

# Una ejecución del grafo que supere esto se corta y el job pasa a FAILED
TIMEOUT_EJECUCION_S = int(os.getenv("JOB_TIMEOUT_SECONDS", "300"))

# Retención en Redis: estado de los jobs y checkpoints de LangGraph (el plan free de Redis Cloud
# tiene 30 MB, así que no se guardan para siempre)
TTL_JOB_S = int(os.getenv("JOB_TTL_SECONDS", str(7 * 24 * 3600)))
TTL_CHECKPOINT_MIN = int(os.getenv("CHECKPOINT_TTL_MINUTES", str(24 * 60)))

# Criterios de criticidad del Human-in-the-loop
UMBRAL_PASOS_CRITICO = int(os.getenv("HITL_UMBRAL_PASOS", "6"))
