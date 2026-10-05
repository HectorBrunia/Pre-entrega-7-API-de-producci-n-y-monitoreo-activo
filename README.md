# Pre-entrega 7: API de producción y monitoreo activo

API REST asíncrona (**FastAPI**) que expone el orquestador multi-agente del Módulo 6 (Supervisor + Investigador + Analista + Validador, con **LangGraph** y **Gemini**). Responde preguntas sobre las políticas internas de *TechCorp*.

| Capacidad | Implementación |
|---|---|
| Endpoints asíncronos | `POST /tasks` encola en **Arq** y devuelve `job_id` en milisegundos (202). El grafo corre en un **worker separado**. |
| Estado en Redis | Cada job es un hash en Redis (`PENDING → RUNNING → WAITING_APPROVAL → DONE / FAILED`). Los checkpoints de LangGraph se guardan con **`AsyncRedisSaver`**. |
| Observabilidad | **LangSmith**: traza por ejecución con cada nodo, cada llamada a Gemini (tokens y costo) y cada tool. |
| Human-in-the-loop | El nodo `aprobacion_humana` hace `interrupt()` en las tareas **críticas**. `POST /tasks/{id}/approve` retoma el grafo desde el checkpoint. |
| Manejo de errores | Una excepción o un timeout llevan el job a `FAILED`. Una tarea periódica (`reaper`) marca `FAILED` los jobs que quedan huérfanos si el worker muere. |

## Arquitectura

```
                     ┌──────────────── Redis (8 / Stack) ────────────────┐
 cliente ──POST /tasks──▶ API FastAPI ──enqueue──▶ cola Arq ──▶ Worker Arq │
    ▲                    │   (async)                              │        │
    │                    ├─ job:{id} (hash de estado) ◀───────────┤        │
    └──GET /tasks/{id}───┘                                        │        │
                                                checkpoints ◀─────┤ grafo LangGraph
                                                (AsyncRedisSaver) │  (M6 + HITL)
                                                                  └──trazas──▶ LangSmith
```

```
mi-api-agente/
├── app/
│   ├── main.py            # FastAPI: POST /tasks · GET /tasks/{id} · GET /tasks · GET /health
│   ├── graph.py           # orquestador del M6 + nodo HITL + AsyncRedisSaver
│   ├── worker.py          # worker de Arq: ejecuta/reanuda el grafo y actualiza el estado en Redis
│   ├── jobs.py            # estado de los jobs en Redis (compare-and-set atómico en Lua)
│   ├── observability.py   # init de LangSmith, @traceable y metadata de traza
│   ├── hitl.py            # criterio de criticidad + interrupt() + POST /tasks/{id}/approve
│   └── config.py          # parámetros desde .env
├── orquestador/           # sistema multi-agente del Módulo 6 (agentes, estado, validador, tools)
├── data/                  # políticas de TechCorp (corpus del RAG)
├── scripts/
│   ├── carga_concurrente.py   # 5 peticiones concurrentes + p50/p95 end-to-end
│   └── metricas_langsmith.py  # costo por ejecución, p95 y tokens por nodo, leídos de LangSmith
├── tests/                 # 29 tests offline (sin API keys, con fakeredis)
├── screenshots/           # evidencia del dashboard de LangSmith
├── docs/grafo.mmd         # diagrama del grafo
├── requirements.txt       # (+ requirements-rag.txt opcional para Pinecone)
├── docker-compose.yml     # redis + api + worker
└── .env.example
```

### El grafo

```mermaid
graph TD;
	__start__([start]) --> supervisor;
	supervisor -.-> investigador;
	supervisor -.-> analista;
	investigador --> validador;
	analista --> validador;
	validador --> supervisor;
	supervisor -. "FINISH, no crítica" .-> sintesis;
	supervisor -. "FINISH, crítica" .-> aprobacion_humana["aprobacion_humana ⏸ interrupt()"];
	supervisor -.-> escalar_humano;
	aprobacion_humana -. aprobado .-> sintesis;
	aprobacion_humana -. rechazado .-> rechazado;
	sintesis --> __end__([end]);
	rechazado --> __end__;
	escalar_humano --> __end__;
```

## Puesta en marcha

Requisitos: Python 3.12+ y un Redis con los módulos **JSON y Search** (Redis 8, Redis Stack o Redis Cloud). El `AsyncRedisSaver` los usa para indexar los checkpoints, así que un Redis 6/7 común no alcanza.

### 1. Instalar

```bash
git clone <este-repo> && cd <este-repo>
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env            # y completá las claves
```

### 2. Inicializar Redis (elegí una opción)

**A. Redis Cloud (gratis, sin instalar nada)**
1. Creá una base *Essentials (Free 30 MB)* en <https://redis.io/try-free/>. Ya trae JSON y Search.
2. En la base, *Connect → Redis CLI* muestra el host, el puerto y el password.
3. En `.env`: `REDIS_URL=redis://default:<password>@<host>:<puerto>/0` (usá `rediss://` si la base tiene TLS).

**B. Docker**
```bash
docker compose up -d redis      # Redis 8 en localhost:6379 → REDIS_URL=redis://localhost:6379/0
```

No hace falta crear índices a mano: el worker ejecuta `AsyncRedisSaver.asetup()` al arrancar.

### 3. Configurar LangSmith

1. Creá una cuenta en <https://smith.langchain.com> → *Settings → API Keys*.
2. En `.env`: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=...` y `LANGSMITH_PROJECT=api-agente-produccion`.
3. **Precio del modelo:** LangSmith calcula el costo con su tabla de precios. Si el modelo de `GEMINI_MODEL` no figura, el costo aparece vacío. Para cargarlo, andá a *Settings → Model Pricing → Add model*, poné el nombre del modelo como *match pattern* y los precios de entrada y salida por 1M de tokens.

### 4. Levantar la API y el worker (dos terminales)

```bash
uvicorn app.main:app --port 8000          # terminal 1: API → http://localhost:8000/docs
arq app.worker.WorkerSettings             # terminal 2: worker
```

Con Docker, `docker compose up --build` levanta los tres servicios.

## Uso de la API

```bash
# 1) Encolar: responde al instante con el job_id
curl -X POST localhost:8000/tasks -H "Content-Type: application/json" \
  -d '{"pregunta": "Un líder de equipo no usó su presupuesto de capacitación el año pasado. Este año quiere hacer un curso de 2300 USD. ¿Cuánto le sobra o le falta?"}'
# → {"job_id": "3f2c...", "status": "PENDING", ...}

# 2) Polling
curl localhost:8000/tasks/3f2c...
# → "status": "WAITING_APPROVAL", "aprobacion_pendiente": {"motivos": ["Impacto económico: ..."],
#    "propuesta": {"hallazgos": [...], "calculo": "1500*2 - 2300 = 700"}}

# 3) Decisión humana (aprobado: false descarta la respuesta)
curl -X POST localhost:8000/tasks/3f2c.../approve -H "Content-Type: application/json" \
  -d '{"aprobado": true, "revisor": "rrhh", "comentario": "monto verificado"}'

# 4) Polling hasta DONE → "resultado": {"respuesta_final": "...", "bitacora": [...]}
```

Para demostrar el camino de error, enviá `"simular_fallo": true` en `POST /tasks`: el worker lanza una excepción y el job termina en `FAILED` con el error, sin quedar colgado en `RUNNING`.

### ¿Cuándo una tarea es "crítica"? (`app/hitl.py`)

El criterio es determinístico, sin LLM, y se evalúa cuando el supervisor decide finalizar:

1. **Impacto económico:** algún dato validado está en USD/$/ARS/EUR, o sea que la respuesta informa dinero que después se paga (horas extra, presupuestos, reintegros). Es la acción con efectos secundarios.
2. **Costo alto:** la ejecución usó al menos `HITL_UMBRAL_PASOS` pasos (6 por defecto) del supervisor, porque hubo reparaciones y la corrida salió cara.

Mientras espera, el grafo queda **detenido en Redis** sin ocupar el worker. Al aprobar, cualquier proceso del worker lo retoma desde el checkpoint, sin repetir las llamadas a los agentes que ya corrieron.

### Estados del job

| Estado | Significado |
|---|---|
| `PENDING` | Encolado, o aprobado y esperando que lo retome un worker |
| `RUNNING` | El worker está ejecutando el grafo |
| `WAITING_APPROVAL` | Pausado en `interrupt()`. Ver `aprobacion_pendiente` y usar `/approve` |
| `DONE` | Terminado. `resultado.respuesta_final` tiene la respuesta (o el rechazo o escalado) |
| `FAILED` | Excepción, timeout (`JOB_TIMEOUT_SECONDS`) o worker caído. Ver `error` |

## Prueba de carga: 5 peticiones concurrentes

Con la API y el worker levantados:

```bash
python scripts/carga_concurrente.py              # aprueba automáticamente las tareas críticas
```

El script lanza las 5 preguntas a la vez (3 de ellas son críticas y pasan por el HITL), hace polling, mide la latencia end-to-end y la etiqueta en LangSmith con `carga:<id>`. Al final imprime el comando para leer las métricas de esa corrida desde LangSmith:

```bash
python scripts/metricas_langsmith.py --corrida <id>
```

Ese script devuelve el **costo por ejecución** (el que LangSmith calcula a partir de los tokens de entrada y salida), la **latencia p50/p95/p99** y los **tokens y latencia por nodo** del grafo. Hace falta porque los gráficos de LangSmith muestran P50 y P99 pero no P95, así que el p95 se calcula sobre los mismos runs que muestra el dashboard.

## Evidencia (`/screenshots`)

| Captura | Qué muestra |
|---|---|
| `01_trazas_proyecto.png` | Lista de trazas del proyecto filtrada por `carga:<id>`, con latencia, tokens y costo de cada ejecución |
| `02_traza_detalle.png` | Árbol de una traza: `tarea_agente → orquestador_multiagente → supervisor / investigador / analista / ...` |
| `03_hitl_thread.png` | Vista *Threads*: la ejecución y la reanudación de un job crítico, agrupadas por `thread_id` |
| `04_costo_por_ejecucion.png` | Dashboard: costo por traza de la corrida concurrente |
| `05_latencia.png` | Dashboard: latencia de la corrida + salida de `metricas_langsmith.py` con el p95 |

### Lectura del dashboard

*(Se completa con los números reales de la corrida.)*

## Tests

```bash
python -m pytest -q
```

Los 29 tests corren offline, sin claves ni Redis real: el grafo usa `InMemorySaver` (misma interfaz que `AsyncRedisSaver`) y el estado de los jobs usa `fakeredis`. Cubren:
- pausa, aprobación y rechazo del HITL;
- `FAILED` ante excepción, timeout y worker caído;
- que una segunda aprobación concurrente sea rechazada con 409;
- los tests originales del orquestador del M6.

## Decisiones de diseño

- **Arq en lugar de `BackgroundTasks`:** el agente corre en otro proceso, así que un pico de tareas no compite con la API por el event loop ni por la CPU. Además, la cola sobrevive a un reinicio de la API y escalar es levantar más workers.
- **Timeout propio con `asyncio.timeout`:** el timeout de Arq cancela la corrutina con `CancelledError`, que no es `Exception` y se saltearía el `except` que marca `FAILED`.
- **Compare-and-set en Lua para `/approve`:** si llegan dos aprobaciones simultáneas, solo una saca el job de `WAITING_APPROVAL`. La otra recibe 409 y el grafo no se reanuda dos veces.
- **TTL en Redis:** los checkpoints (24 h) y los jobs (7 días) expiran, para no llenar los 30 MB del plan gratis de Redis Cloud.
- **Reuso del M6 sin reescribirlo:** los agentes, el validador y el aislamiento de escritura del estado son los mismos. Solo se agregaron el nodo HITL, el campo `aprobacion` del estado y el checkpointer.
