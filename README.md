# TFM-Quantum: Mitigación de Ruido Cuántico mediante Deep Learning

## La idea en 30 segundos

Los ordenadores cuánticos actuales se equivocan en cada operación, como una calculadora imprecisa. Este TFM entrena modelos de IA que **predicen cuánto se equivocará la máquina ANTES de ejecutar el cálculo**, leyendo solo el "parte médico" diario del chip y el diseño del circuito. El trabajo se plantea como una **comparativa rigurosa de 3 modelos** — regresión lineal (Ridge), Random Forest y un Graph Transformer (GEM) — evaluados con protocolo idéntico.

> 📌 **Alcance vigente (jul-2026, acordado con el tutor):** solo el módulo GEM, como comparativa de modelos. El segundo módulo original (REM, corrección del sensor de lectura post-ejecución) queda documentado como trabajo futuro.

> 👤 **¿Primera vez en el proyecto?** Empieza por **[GUIA_TUTOR.md](GUIA_TUTOR.md)** — el recorrido completo sin tecnicismos, con el estado actual y el mapa de lectura.

---

## Descripción técnica

Predicción pre-ejecución del error de circuitos cuánticos en hardware NISQ (IBM Heron r2, 156 qubits) mediante una comparativa de modelos: baselines clásicos de regresión (Ridge, Random Forest) sobre features agregadas vs. un Graph Transformer (GEM) sobre el DAG del circuito con telemetría física por puerta. (El módulo REM post-ejecución del diseño original queda como trabajo futuro.)

---

## Pipeline Arquitectónico

```
Circuito → [GEM] → Δ predicho → IBM QPU (intacto) → ⟨O⟩_mit = ⟨O⟩_noisy − Δ
```

### Módulo 1 — GEM (Gate Error Mitigation)

Predice Δ **antes** de ejecutar el circuito en el QPU. El circuito se representa como un DAG donde cada nodo es una puerta lógica con un vector de 19 features (tipo de puerta, qubits involucrados, error de puerta, longitud de puerta, día del experimento, posición en profundidad...).

Arquitectura: **Graph Transformer** con un nodo virtual QCR (Quantum Circuit Representative) conectado a todos los nodos del grafo. El QCR actúa como resumen global del circuito — su embedding se usa directamente para predecir Δ. Sin message-passing (estilo GTraQEM, no MPNN).

Ventaja diferencial: al predecir Δ antes de la ejecución, el GEM puede actuar como **filtro de calidad del circuito** (rechazar circuitos con Δ demasiado alto) sin gastar tiempo de máquina. Δ es el error TOTAL del circuito (puertas + lectura).

### La comparativa (encuadre del TFM)

| Modelo | Tipo | Ve el grafo | Rol |
|---|---|---|---|
| Ridge | Regresión lineal clásica | No (features agregadas) | Baseline interpretable |
| Random Forest | Ensemble clásico | No (features agregadas) | Baseline fuerte (el mejor en Liao et al., Nature MI 2024) |
| **GEM** | Graph Transformer + QCR | **Sí (DAG completo)** | Modelo propuesto |

Los tres predicen el mismo Δ, con los mismos splits, las mismas métricas y la misma semilla. La pregunta del TFM: *¿aporta la estructura de grafo frente a las features agregadas?*

### Post-procesado

```
⟨O⟩_mit = ⟨O⟩_noisy − Δ
```

### 🔮 Trabajo futuro — Módulo REM (Readout Error Mitigation)

Fuera del alcance vigente. Mitigaría el ruido de lectura **después** de recibir los counts: matrices locales de confusión 2×2 por qubit + **GMRES Matrix-Free** en el subespacio de bitstrings observados (O(N) en memoria). Los esqueletos (`src/rem_model.py`, `configs/rem_config.yaml`) se conservan en el repo.

---

## Estructura del Proyecto

```
TFM-Quantum/
├── configs/
│   ├── gem_config.yaml         # Hiperparámetros GEM (Graph Transformer)
│   └── rem_config.yaml         # Hiperparámetros REM (GNN + GMRES)
├── data/
│   ├── raw/                    # Calibración IBM (ibm_kingston_calib.json, historial T1/T2)
│   └── processed/
│       ├── train/              # Circuitos random, HEA, Trotterized TFIM (días 1–7)
│       ├── val/                # In-distribution validation (día 6 OOD temporal)
│       ├── test/               # In-distribution test (días 8–10)
│       ├── zeroshot_qaoa/      # QAOA — nunca visto en entrenamiento
│       └── zeroshot_qft/       # QFT  — nunca visto en entrenamiento (contribución original)
├── doc/
│   ├── SOTA/
│   │   ├── papers/                   # Los 14 PDFs de la bibliografía analizada
│   │   ├── comparative_analysis.md   # Análisis comparativo de 14 papers vs decisiones TFM
│   │   ├── TFM-SOTA-Parte1.pdf
│   │   └── TFM-SOTA-Parte2.pdf
│   ├── dataset_info.md               # Decisiones de dataset y métricas
│   ├── migracion_heron.md            # Migración forzosa Eagle → Heron (jul 2026) y sus consecuencias
│   ├── src_info.md                   # Documentación de src/
│   ├── notebooks_info.md             # Guía de notebooks
│   └── fallas_y_soluciones.md        # Bugs conocidos y soluciones
├── figures/                    # Gráficas generadas (radar, fidelidad, drift)
├── logs/                       # Artefactos MLflow
├── models/                     # Checkpoints entrenados (.pt)
├── notebooks/
│   ├── 01_eda_ibm_telemetry.ipynb              # EDA + drift step-like
│   ├── 02_quantum_circuit_prototyping.ipynb    # Validación de los 5 tipos de circuito
│   ├── 03_gem_transformer_evaluation.ipynb     # Entrenamiento y métricas del GEM
│   ├── 04_model_comparison.ipynb               # Comparativa Ridge vs RF vs GEM (núcleo del TFM)
│   └── 05_final_pipeline_tradeoffs.ipynb       # Pipeline completo + ablación 70/30/80/20/90/10
├── scripts/                    # Entry-points ejecutables (importan de src/)
│   ├── generate.py             # Genera el dataset (--mini o --full)
│   ├── make_synthetic_calib.py # Calibración sintética Heron (mientras no hay credenciales IBM)
│   ├── make_pipeline_figure.py # Genera figures/pipeline_tfm.png
│   ├── train_gem.py            # Entrena el GEM
│   ├── train_rem.py            # 🔮 Trabajo futuro (REM)
│   └── evaluate.py             # Evalúa el pipeline completo por split
├── src/                        # Código fuente modular (lógica de negocio)
│   ├── config.py               # Rutas, constantes físicas, semillas
│   ├── quantum_gen.py          # CircuitFactory, HardwareTelemetrics, TFMDatasetPipeline
│   ├── dataset.py              # PyG Dataset + DataLoaders (in-dist + zero-shot)
│   ├── gem_model.py            # Graph Transformer con nodo QCR
│   ├── rem_model.py            # GNN local 2×2 + GMRES Matrix-Free
│   ├── train.py                # Entrenamiento desacoplado GEM/REM + MLflow + normalización Δ
│   ├── inference.py            # Pipeline producción: GEM → QPU → REM
│   └── utils.py                # Métricas, GMRES wrapper, encodings
├── .env                        # IBM_TOKEN + IBM_INSTANCE_CRN + IBM_BACKEND_NAME (no versionado)
├── GUIA_TUTOR.md               # Puerta de entrada sin tecnicismos (empieza aquí)
├── .gitignore
├── .pre-commit-config.yaml     # black + flake8 + checks automáticos
├── .dvcignore
├── CLAUDE.md                   # Instrucciones de contexto para Claude Code
├── IDEA_GENERAL_TFM.md         # Descripción completa del TFM en lenguaje claro
├── IDEAS_FUTURAS.md            # 7 ideas de mejora post-TFM
├── ROADMAP.md                  # Estado de implementación por tareas
├── environment.yml             # Dependencias conda (Python 3.10, Qiskit 1.x, PyG)
└── README.md
```

---

## Dataset y Estrategia de Entrenamiento

### Tipos de circuito

| Tipo | Split | Descripción |
|------|-------|-------------|
| Random | Train (40%) | Circuitos aleatorios sobre el coupling map de ibm_kingston |
| HEA | Train (40%) | Hardware-Efficient Ansatz — bloques Ry+CNOT por capa |
| Trotterized TFIM | Train (20%) | Transverse-Field Ising Model — benchmark estándar QEM-Bench |
| QAOA | **Zero-shot A** | Nunca en entrenamiento — reservado para test de generalización |
| QFT | **Zero-shot B** | Nunca en entrenamiento — **contribución original** del TFM |

La proporción 80/20 (random+HEA / TFIM) se ablacionará: 70/30, 80/20, 90/10.

### Split temporal (OOD)

10 días simulados de calibración IBM con **drift step-like** (Hirasaki et al., APL 2023): 2–3 saltos aleatorios de ±15–30% en T1/T2/readout_error a lo largo del timeline. El modelo debe generalizar a días de calibración no vistos.

| Split | Días | Propósito |
|-------|------|-----------|
| Train | 1–5 | Aprendizaje |
| Validation | 6 | Early stopping + tuning |
| Test OOD | 8–10 | Evaluación in-distribution |
| Zero-shot QAOA | — | Generalización a circuito nuevo |
| Zero-shot QFT | — | Generalización a circuito nuevo |

### Drift step-like

```
Error (T1/T2)
│
│  ─────┐
│       │
│       └──────┐
│              │
│              └─────────
└──────────────────────── tiempo (días)
```

Modelo basado en evidencia empírica de hardware IBM real (vs. decay lineal previo, descartado).

---

## Métricas de Evaluación

### GEM

| Métrica | Descripción |
|---------|-------------|
| MAE | Error absoluto medio en predicción de Δ |
| RMSE | Error cuadrático medio en predicción de Δ |
| R² | Coeficiente de determinación |
| Mejora relativa | `(MAE_noisy − MAE_mit) / MAE_noisy` |

Las métricas se calculan con protocolo idéntico para los 3 modelos de la comparativa y se reportan separadas por split: in-distribution / zero-shot QAOA / zero-shot QFT / por día.

*(🔮 Trabajo futuro — métricas REM: Hellinger Fidelity, TVD, ratio HF_mit/HF_noisy.)*

---

## MLOps

- **DVC**: versiona datasets y modelos grandes. Remote local en `~/TFM/dvc-remote/`
- **MLflow**: tracking de experimentos, métricas y checkpoints
- **pre-commit**: `black` (formateo) + `flake8` (linting) se ejecutan automáticamente en cada commit
- **configs/**: YAMLs de hiperparámetros para reproducibilidad total

---

## Instalación

```bash
# 1. Crear el entorno conda
conda env create -f environment.yml
conda activate tfm

# 2. Instalar hooks de pre-commit
pre-commit install

# 3. Configurar credenciales IBM (no commitear)
cp .env.example .env   # añadir IBM_TOKEN e IBM_INSTANCE_CRN
```

## Uso

```bash
# Generar dataset pequeño (150 muestras, validación rápida)
conda run -n tfm python scripts/generate.py --mini

# Generar dataset completo (5000 muestras, 10 días)
conda run -n tfm python scripts/generate.py --full

# Versionar con DVC
dvc add data/processed/
dvc push

# Entrenar
conda run -n tfm python scripts/train_gem.py
conda run -n tfm python scripts/train_rem.py

# Evaluar (todos los splits)
conda run -n tfm python scripts/evaluate.py \
    --gem models/gem_best.pt \
    --rem models/rem_best.pt \
    --split all
```

---

## Hardware objetivo

- **Dispositivo**: `ibm_kingston` — familia **Heron r2**, 156 qubits, puerta 2q nativa **CZ**. Accesible en Open Plan.
  - El backend original del proyecto era `ibm_kyiv` (Eagle r3); IBM retiró toda la familia Eagle entre 2023 y abril de 2026, forzando la migración.
- **Canal IBM**: `ibm_quantum_platform` (nuevo, no el deprecated `ibm_quantum`)
- **Validación real**: 20–30 circuitos en hardware real para medir la brecha simulation-to-real (TAREA 6)

---

## Referencias clave

- **GTraQEM** — Bao et al., ICLR 2025: Graph Transformer con nodo QCR (baseline arquitectónico del GEM)
- **QEMFormer + QEM-Bench** — Bao et al., ICML 2025: el SOTA a superar; benchmark estándar del campo
- **M3** — Nation et al. (IBM), PRX Quantum 2021: mitigación de readout matrix-free (fundamento del REM)
- **Concept Drift** — Hirasaki et al., APL 2023 (evidencia física) + Huo et al., SIGMETRICS 2026 (evidencia operacional)
- **Q-Cluster** — Patil et al., IEEE QCE 2025: paradigma no supervisado de contraste
- Lista completa de 14 papers en `doc/SOTA/comparative_analysis.md`
