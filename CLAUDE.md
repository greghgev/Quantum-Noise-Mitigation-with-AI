# CLAUDE.md — Memoria de Sesión: TFM-Quantum

> Lee este archivo al inicio de cada sesión. Contiene el contexto completo del proyecto.

## INSTRUCCIONES PERMANENTES

- **Ignorar archivos `.TODO`**: No leer, no referenciar, no tener en cuenta ningún archivo con extensión `.TODO` del proyecto (ej. `Ideas.TODO`, `TODO`). Esta instrucción está en vigor hasta que el usuario la revoque explícitamente.

---

## 1. Objetivo del TFM y enfoque de mitigación de ruido

**Título:** Mitigación de Ruido Cuántico en Hardware NISQ mediante Deep Learning

**Problema:** Los procesadores cuánticos actuales (era NISQ) introducen dos tipos de ruido que degradan los resultados:
1. **Ruido dinámico de puertas** — acumulación térmica y de despolarización durante la ejecución del circuito.
2. **Ruido de lectura (readout)** — errores del sensor láser en la medición final.

**⚠️ ALCANCE VIGENTE (jul-2026, acordado con el tutor): SOLO GEM, como comparativa de modelos.**
El REM es TRABAJO FUTURO (ficheros conservados como esqueleto; ver IDEAS_FUTURAS.md). Δ es ahora
el **error TOTAL** (puertas + readout) — el desacoplamiento de ruidos queda anulado.

**Solución (alcance vigente):**

- **GEM (Gate Error Mitigation):** Graph Transformer que ingiere el DAG del circuito y la telemetría del chip, y predice un valor continuo de desviación **Δ** (error total) ANTES de ejecutar. El circuito se ejecuta *intacto* en IBM (no se modifica — ver "Trampa del Transpilador" en sección 6).
- **Comparativa de 3 modelos** (encuadre del TFM): **Ridge** (regresión lineal clásica) + **Random Forest** (mejor baseline según Liao 2024) + **GEM** (Graph Transformer). Baselines tabulares sobre features agregadas del circuito; protocolo de evaluación IDÉNTICO para los tres. Lo que se valora es la comparativa rigurosa, no que un modelo sea excelente.
- 🔮 *Trabajo futuro — REM (Readout Error Mitigation):* GNN que predice matrices de confusión locales 2×2 por qubit + corrección Matrix-Free (GMRES en subespacio de bitstrings observados, O(N)).

**Flujo de inferencia (alcance vigente):**
```
Circuito usuario → GEM predice Δ (error total) → Ejecución en IBM QPU → ⟨O⟩_mitigado = ⟨O⟩_ruido - Δ
```

---

## 2. Stack tecnológico y versiones clave

| Componente | Versión / Notas |
|---|---|
| Python | 3.10 |
| Conda env | `tfm` (ver `environment.yml`) |
| Qiskit | `2.4.1` (migrado desde 1.0.2 — APIs deprecated sustituidas) |
| qiskit-aer | `0.17.2` |
| qiskit-ibm-runtime | `0.47.0` |
| PyTorch | latest (sin pin de versión) |
| torch-geometric | latest (canal `pyg`) |
| SciPy | sin pin — crítico para GMRES (`scipy.sparse.linalg`) |
| MLflow | latest |
| Weights & Biases | latest |
| python-dotenv | latest — carga `.env` |

**Entorno de ejecución:** WSL2 (Linux 6.6.x), sin GPU confirmada en local.

---

## 3. Estructura del repositorio

```
TFM-Quantum/
├── configs/
│   ├── gem_config.yaml         # Hiperparámetros GEM (Graph Transformer + QCR)
│   └── rem_config.yaml         # Hiperparámetros REM (GNN + GMRES)
├── data/
│   ├── raw/                    # Calibración IBM (ibm_kingston_calib.json) — generado en runtime
│   └── processed/              # Grafos .pt listos para PyTorch Geometric
│       ├── train/              # Random, HEA, TFIM (días 1–5)
│       ├── val/                # In-distribution validation (día 6)
│       ├── test/               # In-distribution test (días 8–10)
│       ├── zeroshot_qaoa/      # QAOA — nunca en entrenamiento
│       └── zeroshot_qft/       # QFT  — nunca en entrenamiento
├── doc/
│   ├── dataset_info.md         # Especificaciones y justificaciones del dataset
│   ├── fallas_y_soluciones.md  # Decisiones arquitectónicas críticas (leer obligatorio)
│   ├── notebooks_info.md       # Guía de notebooks
│   ├── src_info.md             # Documentación de módulos src/
│   └── SOTA/
│       ├── papers/                  # Los 14 PDFs de la bibliografía analizada
│       ├── comparative_analysis.md  # Análisis comparativo 14 papers vs TFM
│       ├── TFM-SOTA-Parte1.pdf
│       └── TFM-SOTA-Parte2.pdf
├── figures/            # pipeline_tfm.png (diagrama divulgativo) + gráficas generadas
├── logs/               # Artefactos MLflow (vacío, se genera)
├── models/             # Pesos .pt y checkpoints (vacío, se genera)
├── notebooks/          # 01 ✅ ejecutado (EDA divulgativa) · 02-05 y main.ipynb vacíos (placeholders)
├── scripts/            # Entry-points ejecutables (importan de src/, no contienen lógica)
│   ├── generate.py     # Genera el dataset (--mini 150 muestras | --full 5000)
│   ├── make_synthetic_calib.py  # Calibración sintética Heron mientras no hay credenciales IBM
│   ├── make_pipeline_figure.py  # Genera figures/pipeline_tfm.png (diagrama divulgativo)
│   ├── train_gem.py    # Entrena el GEM con gem_config.yaml
│   ├── train_rem.py    # Entrena el REM con rem_config.yaml
│   └── evaluate.py     # Evalúa pipeline completo (--split test|zeroshot_qaoa|zeroshot_qft|all)
├── src/                # Código fuente de producción (lógica real — importado por scripts/)
│   ├── config.py       ✅ COMPLETO
│   ├── quantum_gen.py  ✅ COMPLETO (TAREA 1 + 3 + 3b + 8)
│   ├── dataset.py      ❌ VACÍO
│   ├── gem_model.py    ❌ VACÍO
│   ├── rem_model.py    🔮 TRABAJO FUTURO (REM fuera de alcance)
│   ├── train.py        ⚠️  ESQUELETO (solo docstring)
│   ├── utils.py        ⚠️  ESQUELETO (solo docstring)
│   └── inference.py    🔮 TRABAJO FUTURO (pipeline GEM→QPU→REM)
├── tests/
│   └── test_quantum_gen.py  ✅ 37/37 PASSED
├── .dvc/               # Metadata DVC (versionado)
├── .dvcignore
├── .env                # IBM_TOKEN + IBM_INSTANCE_CRN + IBM_BACKEND_NAME (no subir a git)
├── .gitignore
├── .pre-commit-config.yaml  # black + flake8 + checks automáticos (instalados con pre-commit install)
├── ROADMAP.md          # Tareas pendientes y estado del proyecto
├── GUIA_TUTOR.md       # Puerta de entrada sin tecnicismos: pipeline, estado, mapa de lectura, glosario
├── IDEA_GENERAL_TFM.md # Explicación conceptual del TFM: pipeline, diferenciadores, métricas, dataset, decisiones
├── IDEAS_FUTURAS.md    # 7 extensiones y mejoras potenciales para trabajo futuro
├── environment.yml
└── CLAUDE.md
```

---

## 4. Estado actual de implementación

### Completos
- **`src/config.py`** — Carga `.env`, define rutas (`BASE_DIR`, `RAW_DATA_PATH`, `PROCESSED_DATA_PATH`, `MODELS_PATH`), backend (`BACKEND_NAME=ibm_kingston`, `BASIS_GATES=[cz,id,rz,sx,x]`), constantes físicas (`MAX_QUBITS=15`, `MAX_DEPTH=50`, `LABEL_SHOTS=4096` para etiquetas Δ, `TRAIN_SHOTS=1024` para inputs REM, `SEED=42`), función `set_global_seeds()`.

- **`src/quantum_gen.py`** ✅ COMPLETO (TAREA 1 + TAREA 3 + migración Heron) — 4 clases:
  - `CircuitFactory`: genera circuitos random, QFT, QAOA (ZZ por arista), HEA, Trotterized TFIM; transpila a puertas nativas IBM Heron (`cz` como 2q).
  - `HardwareTelemetrics`: descarga/cachea calibración de `ibm_kingston`; drift step-like pre-computado (2–3 saltos ±15–30% en 10 días); `build_noise_model_for_day()` construye NoiseModel por día (despolarización 1q/2q + readout simétrico).
  - `QuantumGraphExtractor`: circuito + Δ + telemetría → objeto `Data` PyG con 19 dims por nodo, `y = Δ` escalar, `ideal_probs` y `noisy_probs` adjuntos.
  - `TFMDatasetPipeline`: orquestador con `run()`, routing por split (QAOA/QFT solo zero-shot), seeds independientes por split (sin leakage, TAREA 8), simulación Aer ruidosa (`method='statevector'`, `LABEL_SHOTS`).
- **`tests/test_quantum_gen.py`** ✅ — 37/37 PASSED (calibración falsa en tmp_path, sin conexión IBM; incluye test de regresión anti-leakage).

### Pendientes (alcance vigente: GEM + comparativa)
- **`src/dataset.py`** — Clases `Dataset` de `torch_geometric.data` para cargar los `.pt` en DataLoaders.
- **`src/features.py`** (nuevo) — Features agregadas del circuito para los baselines tabulares.
- **`src/baselines.py`** (nuevo) — Ridge + Random Forest (sklearn).
- **`src/gem_model.py`** — Arquitectura Graph Transformer (regresor continuo → predice Δ).
- **`src/train.py`** — Entrenamiento de los 3 modelos con protocolo idéntico + MLflow/W&B.
- **`src/utils.py`** — Métricas GEM (MAE, RMSE, R², mejora relativa), encodings.

### 🔮 Trabajo futuro (fuera del alcance vigente — NO implementar)
- **`src/rem_model.py`** — GNN + Matrix-Free GMRES (módulo REM completo).
- **`src/inference.py`** — Pipeline producción GEM → QPU → REM.
- **Notebooks 02–05** — Pendientes de implementar (0 bytes). `main.ipynb` es un placeholder vacío (era de pruebas).
  - **Notebook 01** ✅ COMPLETO — EDA divulgativa ejecutada (circuito dibujado, DAG coloreado, heatmap de drift, franja de shot-noise, conclusiones sobre outputs reales).

### Datos actuales
- **`data/raw/`** — `ibm_kingston_calib.json` sintético [SUPOSICIÓN] (regenerar con calibración real al tener credenciales).
- **`data/processed/`** — mini-dataset Heron: 178 muestras con los fixes de TAREA 8 (sin leakage, verificado), versionado en DVC. Señal Δ: train ratio señal/ruido 2.7×; zero-shot aún bajo el suelo de shot-noise a 4096 shots (subir `LABEL_SHOTS` en el full run).

---

## 5. Convenciones de código

- **Estilo:** PEP8 estricto (`black` + `flake8` vía pre-commit hooks).
- **Diseño:** SOLID + POO. Sin funciones sueltas de más de 20 líneas en notebooks (extraer a `src/`).
- **Sin hardcoding:** Todo parámetro, ruta y token pasa por `src/config.py` o `.env`. Nunca inline.
- **Semillas:** Siempre inicializar con `set_global_seeds()` al arrancar notebooks y scripts.
- **Formato de datos:** Grafos nativos `.pt` (PyTorch Geometric `Data`). Prohibido generar CSV/Parquet intermedios (cuello de botella CPU).
- **Ground Truth:** Extraer con `Statevector.from_instruction(qc)` — analítico, no por shots.
- **Notebooks:** La primera celda de código llama a `set_global_seeds()`. Cada sección va precedida por Markdown con fundamento matemático. Los `Run All` deben ser reproducibles.

---

## 6. Decisiones técnicas importantes

### MLOps: DVC + pre-commit + scripts/

- **DVC** versiona datasets y modelos grandes (`.pt`, `ibm_kingston_calib.json`). Remote local en `~/TFM/dvc-remote/`. Inicializado: `dvc init` ya ejecutado. Comandos habituales: `dvc add data/processed/` → `dvc push`.
- **pre-commit hooks** instalados (`pre-commit install` ya ejecutado). En cada `git commit` se ejecutan automáticamente: `black` (formateo), `flake8 --max-line-length=100` (linting), trailing whitespace, YAML check, detección de claves privadas, y bloqueo de ficheros >500 KB.
- **`scripts/` vs `src/`**: `src/` contiene toda la lógica (clases, funciones, modelos). `scripts/` son entry-points delgados que importan de `src/` y orquestan el pipeline. **Nunca añadir lógica de negocio directamente en `scripts/`**.
- **Configs YAML**: toda configuración de hiperparámetros va en `configs/gem_config.yaml` y `configs/rem_config.yaml`. Los scripts los leen con `yaml.safe_load`. Nunca hardcodear hiperparámetros en código.

### Credenciales IBM
- **Canal:** `"ibm_quantum_platform"` — es el canal moderno (Open Plan + Plan de pago). **NO** usar `"ibm_quantum"` (legacy, deprecado).
- **Backend objetivo:** `ibm_kingston` (familia **Heron r2**, 156 qubits, accesible en Open Plan). Configurable vía `IBM_BACKEND_NAME` en `.env`.
  - **Historia:** el backend original era `ibm_kyiv` (Eagle r3, 127q, Heavy-Hex), retirado por IBM el 18-abr-2025. Toda la familia Eagle está retirada desde abr-2026 — no existe alternativa Eagle.
- **Token:** variable `IBM_TOKEN` en `.env`.
- **CRN de instancia:** variable `IBM_INSTANCE_CRN` en `.env`. Se pasa como `instance=` al construir `QiskitRuntimeService`.
  - ⚠️ Estado jul-2026: cuenta IBM antigua sin instancias (expiró); cuenta nueva pendiente de verificación por IBM (error tarjeta → email a verify@us.ibm.com enviado). Hasta entonces se usa calibración sintética.
- **Caché de calibración:** `data/raw/ibm_kingston_calib.json` — si existe, se carga offline; si no, se descarga de IBM y se guarda. La actual es **SINTÉTICA** (generada por `scripts/make_synthetic_calib.py` con valores plausibles Heron r2 marcados [SUPOSICIÓN]) — borrarla y relanzar para descargar la real cuando haya credenciales.

### Decisión arquitectónica GEM: "Trampa del Transpilador" + nodo virtual QCR
El Graph Transformer **no** reescribe ni modifica el circuito. Solo predice Δ (desviación continua). El circuito se ejecuta intacto para no destruir la unitaria del algoritmo del usuario.

**Nodo virtual QCR (Quantum Circuit Representative):** nodo ficticio añadido al grafo con aristas directas a todos los nodos reales. Durante la atención del Transformer, todos los nodos "hablan" con el QCR simultáneamente y él les responde a todos. Al final del forward pass, el embedding del QCR se usa para predecir Δ (en lugar de average pooling sobre todos los nodos). Permite contexto global sin message-passing. Respaldado empíricamente por GTraQEM [Bao et al., ICLR 2025].

**Δ global vs. Δ por observable — limitación conocida:** el GEM predice un único Δ escalar para todo el circuito. GTraQEM predice un Δ por cada observable medido (⟨Z_i⟩, ⟨Z_iZ_j⟩, etc.), lo que es más preciso cuando distintos observables tienen sensibilidades de ruido distintas. La simplificación del TFM reduce la complejidad de implementación a costa de expresividad. Se documenta en `IDEA_GENERAL_TFM.md` como limitación y en `IDEAS_FUTURAS.md` como extensión posible.

### Decisión arquitectónica REM: "Matrix-Free Subspace Inversion"
La GNN predice matrices 2×2 locales por qubit (y opcionalmente 4×4 para vecinos con crosstalk). **Nunca** se construye la matriz global 2^n × 2^n. El solver GMRES de SciPy opera en el subespacio de bitstrings observados (máx. 1024 con TRAIN_SHOTS=1024). Complejidad: O(N).

### Split temporal OOD (Concept Drift)
- Train: días 1–5 | Val: día 6 | Gap sin datos: día 7 | Test OOD: días 8–10.
- **Modelo de drift: step-like (no lineal).** Hirasaki et al. (APL 2023) demuestra empíricamente que las tasas de error en IBM cambian de forma escalonada, no gradual. El simulador aplica 2–3 saltos aleatorios a lo largo de los 10 días donde T1/T2 y readout_error cambian bruscamente (±15–30% por salto). El decay lineal previo (`1.0 - day_index * 0.02`) está **eliminado** — era factualmente incorrecto.

### Composición del dataset
- **~80% circuitos aleatorios** (Random Unitary) — generalizan la física del ruido. La proporción exacta (70/30, 80/20, 90/10 random/estructurados) se **ablacionará** durante la fase de entrenamiento y se elegirá según rendimiento en validación. Sin precedente empírico fijo en la literatura.
- **~20% estructurados (en entrenamiento):**
  - **HEA** (Hardware-Efficient Ansatz, capas Ry+CX construidas manualmente — `TwoLocal` deprecated en Qiskit 2.x, `reps=2`) — sustituye a Grover, profundidad O(reps×n). Benchmark estándar VQE.
  - **Trotterized TFIM** (simulación de Hamiltoniano Ising con campo transverso) — benchmark estándar de QEM-Bench (QEMFormer, ICML 2025). Permite comparación directa con el SOTA.
- **Zero-shot (NUNCA en entrenamiento):** QAOA y QFT se reservan exclusivamente para evaluación zero-shot. Si el modelo generaliza a ellos sin haberlos visto, demuestra que aprendió física de ruido, no patrones de circuitos. QFT es **contribución original**: ningún paper de ML-QEM lo evalúa sistemáticamente.
- Qubits por circuito: 5–15 (límite RAM: Statevector de >15 qubits → OOM en 16 GB).
- Profundidad: 5–50 capas.

### Puertas nativas IBM Heron
`['cz', 'id', 'rz', 'sx', 'x']` — `CircuitFactory` transpila todos los circuitos a este conjunto base (definido en `src/config.py` como `BASIS_GATES`).

**Diferencia clave vs. Eagle:** la puerta 2-qubit nativa es **CZ** (controlled-Z, simétrica), no CX. Los generadores construyen la lógica con CX (convención estándar) y el transpilador la convierte. En el one-hot del feature vector, la posición 0 es `cz`. Para CZ no existe control/target físico — por convención control=qargs[0], target=qargs[1].

### Métricas de evaluación

**GEM** (predicción de error en valores esperados):

| Métrica | Implementación |
|---|---|
| MAE sobre ⟨O⟩ | `utils.py` |
| RMSE sobre ⟨O⟩ | `utils.py` |
| R² | `utils.py` |
| Mejora relativa vs. sin mitigar | `utils.py` |

Las métricas se calculan con protocolo IDÉNTICO para los 3 modelos de la comparativa (Ridge, RF, GEM) y se reportan separadas por: in-distribution / zero-shot QAOA / zero-shot QFT / por día del split temporal.

🔮 *Trabajo futuro — métricas REM (si se retoma):* Hellinger Fidelity, TVD, ratio HF_mit/HF_noisy.

### Vector de features por nodo (grafo) — 19 dimensiones objetivo

| # | Feature | Dims | Estado |
|---|---|---|---|
| 1 | One-Hot tipo de puerta (`cz`,`id`,`rz`,`sx`,`x`,`measure`) | 6 | ✅ Implementado |
| 2 | Ángulos paramétricos θ, φ, λ (padding a 0.0 si no aplica) | 3 | ✅ Implementado |
| 3 | T1, T2, readout_error del **qubit control** | 3 | ✅ Implementado |
| 4 | gate_error del qubit control (tasa empírica de error de la puerta) | 1 | ✅ Implementado |
| 5 | gate_length del qubit control (duración del pulso en segundos) | 1 | ✅ Implementado |
| 6 | T1, T2, readout_error del **qubit target** (zero-pad [0,0,0] para puertas 1q) | 3 | ✅ Implementado (zero-pad correcto) |
| 7 | day_index normalizado (`day / 10.0`) — señal explícita de Concept Drift | 1 | ✅ Implementado |
| 8 | depth_position normalizado (`pos / depth_total`) — decoherencia acumulada | 1 | ✅ Implementado |

**Total: 19 dims — diseño completo implementado y verificado por tests (TAREA 1).** Nota Heron: CZ es simétrica; "control"=qargs[0], "target"=qargs[1] por convención del transpilador.

Nota: `gate_error` y `gate_length` son estándar en la literatura (QuESt [Wang et al., 2022, ICCAD], GTraQEM [Bao et al., 2025, ICLR]) y disponibles vía `backend.properties()` de IBM.

---

## 7. Estado de bugs y limitaciones conocidas

**Los 5 bugs de TAREA 1 están corregidos** (ver ROADMAP.md). Limitaciones activas:

1. **Calibración sintética (no real):** `data/raw/ibm_kingston_calib.json` está generado por `scripts/make_synthetic_calib.py` con valores plausibles Heron r2 marcados [SUPOSICIÓN] (20 qubits, cadena lineal). Cuando la cuenta IBM esté verificada: borrar el JSON, relanzar `HardwareTelemetrics` para descargar calibración real, y regenerar el dataset.

2. **Noise model solo despolarización:** `thermal_relaxation_error` + `depolarizing_error` combinados producen "Kraus is empty" en Aer 0.17. Se usa solo despolarización; el efecto T1/T2 se modela implícitamente vía drift del gate_error. Documentado en `build_noise_model_for_day()`.

3. **[SUPOSICIÓN] Error CZ ≈ 5× error 1q promedio** de los dos qubits del par — ratio típico NISQ, no dado por la API de calibración. Ver `_populate_noise_model()`.

4. **Readout simétrico:** P(0→1) = P(1→0). IBM real es asimétrico — mejora planificada en TAREA 7a (feature vector pasaría a 20 dims).

5. **`coupling_map` serialización tuplas vs. listas:** `backend.coupling_map.get_edges()` devuelve tuplas; al cargar desde JSON se recuperan como listas anidadas. Sin impacto detectado (el código itera con desempaquetado), pero verificar en la primera descarga real de Heron.

---

## 8. Referencias Clave

> **Leyenda de veracidad** (verificado con DOI real en esta sesión):
> - ✅ `[REVISADO]` — Publicado en revista/conferencia peer-reviewed. DOI verificado. Citar sin reservas.
> - ⚠️ `[PREPRINT]` — Solo arXiv. Sin revisión externa. Si se cita, indicar explícitamente "preprint".

### Baseline principal del TFM ✅ [REVISADO — ICLR 2025]
- **GTraQEM** — Bao et al., 2025, ICLR. "Beyond Circuit Connections: A Non-Message Passing Graph Transformer Approach for Quantum Error Mitigation." [PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/c276c3303c0723c83a43b95a44a1fcbf-Paper-Conference.pdf)
  El trabajo más cercano al TFM. Graph Transformer sin paso de mensajes + nodo virtual QCR (Quantum Circuit Representative) para resumen global. **Leer antes de implementar `gem_model.py`.**

### ML-QEM — Referencia central (SOTA Sección 3) ✅ [REVISADO — Nature Machine Intelligence 2024]
- **Liao, Wang, Sitdikov, Salcedo, Seif, Minev — Nature Machine Intelligence 6, 1478-1486, 2024.** "Machine Learning for Practical Quantum Error Mitigation." DOI: 10.1038/s42256-024-00927-2
  GNNs/MLPs reducen coste de ZNE en 2× en hasta 100 qubits reales de IBM sin perder precisión.

### Contraste arquitectónico — mitigación sin ground truth ✅ [REVISADO — npj Quantum Information 2025]
- **Liao, Zhu, Chiribella, Yang — npj Quantum Information 11, art. 8, 2025.** "Noise-agnostic quantum error mitigation with data augmented neural models." DOI: 10.1038/s41534-025-00960-y
  Mitiga sin `Statevector` como ground truth — usa data augmentation cuántica. Relevante para la sección de limitaciones del TFM: nuestro pipeline requiere ground truth analítico.

### Estudio sistemático de arquitecturas DL para QEM ⚠️ [PREPRINT — arXiv:2601.14226, Quantinuum, sin publicar]
- **Placidi et al. (Quantinuum) — arXiv:2601.14226, Ene 2026.** "Deep Learning Approaches to Quantum Error Mitigation."
  Comparativa sistemática MLP/RNN/Transformer/Perceiver en IBM QPUs reales (ibm_algiers, ibm_hanoi). Pre-entrenamiento en simulador → fine-tuning en hardware. **Ablation study clave:** P_noisy es el feature dominante; T1/T2/gate_error/readout son secundarios pero útiles → justifica nuestro vector de 19 dims. Transfer entre dispositivos funciona si las características de hardware son comparables. **Diferencia arquitectónica con GEM:** ellos mitigan distribuciones completas Pnoisy→Pideal; nuestro GEM predice escalar Δ (pre-ejecución, sin P_noisy). *(Institución: Quantinuum. Sin DOI de revista en el momento de verificación.)*

### GNN para predicción de expectation values con feature vector comparable ✅ [REVISADO — Frontiers of Physics 2026]
- **Liu et al. (Southeast University) — Frontiers of Physics 21(6), 063201, 2026.** "Output Prediction of Quantum Circuits based on Graph Neural Networks." DOI: 10.15302/frontphys.2026.063201
  GNN sobre DAG del circuito para predecir expectation values bajo ruido. Vector de nodo 31-dimensional (vs. 19 dims del TFM) — ver desglose en §6. GNN supera a CNN especialmente cuando crece profundidad/qubits. Extrapolación: entrenado en 3-7 qubits, aplicado a 16 qubits con R²≥0.92. Hardware IBM (Perth, Lagos, Nairobi, Jakarta). **Útil para:** justificar GNN > CNN (§4.2.1) y comparar opciones de feature engineering.

### Nuevo baseline más fuerte — QEMFormer + QEM-Bench ✅ [REVISADO — ICML 2025, PMLR v267 pp. 2953-2967] ⭐ SUPERA A GTraQEM
- **Bao, Zhong, Ye, Tang, Yan (Shanghai Jiao Tong U.) — ICML 2025, PMLR v267.** "QEM-Bench: Benchmarking Learning-based Quantum Error Mitigation and QEMFormer as a Multi-ranged Context Learning Baseline." *(Código prometido en GitHub, no disponible aún al momento de verificación.)*
  **QEM-Bench**: 22 datasets estandarizados (9 estándar + 9 zero-shot + 4 large-scale IBM real). Large-scale usa `ibm_kyiv` (50q) y `ibm_brisbane` (63q) — mismo hardware que el TFM. Tipos: Trotterized TFIM, Random, QAOA. Baselines incluidos: GNN, MLP, ZNE, CDR, RF, GTranQEM.
  **QEMFormer**: Arquitectura dos ramas — Rama MLP (contexto corto, mean pooling global) + Rama Graph Transformer (topología, largo alcance). Node features: gate type one-hot + qubit multi-hot + ángulos. Global features: estadísticas del circuito (8 dims) + expectation values Pauli (3×N_q) + P_noisy (EV ruidoso ya medido). **QEMFormer > GTranQEM en todos los settings.** ibm_kyiv: MAE=0.018, RMSE=0.026.
  **Argumento diferenciador del TFM vs. QEMFormer:** (1) QEMFormer requiere P_noisy como input — necesita ejecutar el circuito antes de mitigar; nuestro GEM opera PRE-ejecución con solo telemetría del hardware → paradigma más eficiente operacionalmente. (2) QEMFormer usa multi-hot qubit encoding (no generalizable a N arbitrario); nuestro vector no codifica identidad de qubit → más generalizable. (3) QEMFormer no modela Concept Drift temporal; nuestro day_index es feature explícito. (4) El TFM separa ruido de puertas (GEM) y de lectura (REM) — QEMFormer los confunde en un solo predictor.
  **Nota GTraQEM:** GTraQEM sigue siendo referencia válida de arquitectura base (Graph Transformer + nodo virtual QCR). QEMFormer es el nuevo techo de rendimiento a superar.

---

## Referencias — Módulo REM y Readout Error Mitigation

### Fundamento directo del módulo REM — M3 ✅ [REVISADO — PRX Quantum 2021, IBM Research] ⭐
- **Nation, Kang, Sundaresan, Gambetta (IBM Research) — PRX Quantum 2, 040326, 2021.** "Scalable mitigation of measurement errors on quantum computers." DOI: 10.1103/PRXQuantum.2.040326
  M3 (Matrix-free Measurement Mitigation): solver iterativo precondicionado que opera en el **subespacio de bitstrings ruidosos observados**, sin construir la matriz de asignación global 2^n × 2^n. Converge en O(1) pasos y usa "orders of magnitude less memory than direct factorization". De investigadores IBM (Gambetta = Chief Quantum Officer). **Relación con TFM:** el módulo REM es una extensión directa de M3: M3 usa matrices de confusión fijas por qubit; nuestro REM las predice dinámicamente via GNN y luego aplica el mismo solver GMRES en subespacio → más adaptativo al ruido del día. **Citar obligatoriamente en §3 y en la descripción del módulo REM.**

### Primera aplicación de deep learning a QREM ✅ [REVISADO — New Journal of Physics 2022]
- **Kim, Oh, Chong, Hwang, Park — New Journal of Physics 24, 073009, 2022.** "Quantum readout error mitigation via deep learning." DOI: 10.1088/1367-2630/ac7b3d
  Red neuronal entrenada con mediciones de circuitos de un solo qubit para corregir ruido de lectura no-lineal en IBM 5-qubit. Supera a la inversión lineal estándar en MSE, KL-divergence e infidelidad. **Diferencia con REM del TFM:** ellos predicen corrección directa (caja negra); nuestro REM predice matrices de confusión locales 2×2 por qubit (interpretable) + solver GMRES → escalable a N qubits. Útil como baseline de comparación para el módulo REM.

### Escalabilidad QREM via independencia condicional + transfer learning ✅ [REVISADO — ML: Science and Technology 2023]
- **Lee & Park — Machine Learning: Science and Technology 4, 045051, 2023.** "Scalable quantum measurement error mitigation via conditional independence and transfer learning." DOI: 10.1088/2632-2153/ad1007
  Explota la independencia condicional entre qubits no vecinos para reducción **exponencial** del tamaño de red neuronal. Transfer learning añade factor constante de speedup. Validado en IBM 7q y 13q. **Conecta con la arquitectura REM del TFM:** la elección de matrices 2×2 locales por qubit en lugar de la matriz global 2^n × 2^n tiene fundamento en este resultado — los errores de lectura entre qubits distantes son condicionalmente independientes.

---

## Referencias — Paradigma pre-ejecución relacionado (selección/ranking, NO mitigación)

### Herramienta nativa de IBM para pre-ejecución ✅ [REVISADO — PRX Quantum 2023] — related work obligatorio
- **Nation & Treinish (IBM) — PRX Quantum 4, 010327, 2023.** "Suppressing Quantum Circuit Errors Due to System Variability" (herramienta: **mapomatic**, qiskit-community).
  Puntúa layouts isomorfos ANTES de ejecutar con la calibración del día (heurística: producto de fidelidades; sin ML) y elige el mejor. Reconoce como limitaciones: crosstalk ignorado y calibración desactualizada. **Diferencia con el GEM:** score heurístico relativo para ELEGIR vs. predicción aprendida de Δ para CORREGIR. Citar en §3: demuestra que IBM ve valor en el pre-ejecución pero solo lo cubre heurísticamente.

### Ranking ML pre-ejecución ✅ [REVISADO — Quantum 8, 1542, 2024] — el precedente más cercano a la "Capacidad 1" del GEM
- **Hartnett, Barbosa, Mundada, Hush et al. (Q-CTRL) — Quantum 8, 1542, nov-2024.** "Learning to rank quantum circuits for hardware-optimized performance enhancement." arXiv:2404.06535.
  ML entrenado en hardware IBM real que rankea circuitos equivalentes pre-ejecución (selección de layout): fidelidades de layouts equivalentes difieren hasta 10×; su modelo mejora la selección 1.8× vs. heurística. **Diferencias con el GEM:** (1) rankea variantes del mismo circuito, no arbitrarios; (2) predice orden, no la magnitud Δ (no permite corregir); (3) modelo fenomenológico, no grafo+telemetría; (4) sin drift temporal. Ver análisis completo en `doc/SOTA/comparative_analysis.md` §13.

## Referencias — Concept Drift y Variabilidad Temporal

### Evidencia empírica directa del Concept Drift cuántico ✅ [REVISADO — Applied Physics Letters 2023]
- **Hirasaki, Daimon, Itoko, Kanazawa, Saitoh — Applied Physics Letters 123, 184002, 2023.** "Detection of temporal fluctuation in superconducting qubits for quantum error mitigation." DOI: 10.1063/5.0166739 *(Kanazawa = IBM Research, desarrollador Qiskit)*
  Observan **cambios abruptos tipo step en las tasas de error** de qubits superconductores, que se repiten múltiples veces y persisten durante varios minutos cada uno. No es drift gradual — son saltos discretos. Proponen post-selection basado en correlaciones entre varianza del output y error elevado. **Relevancia directa para el TFM:** es la evidencia empírica que motiva el `day_index` como feature explícito del Concept Drift. Citar en §3 al justificar la estrategia de split temporal OOD.

### Variabilidad operacional temporal/espacial ✅ [REVISADO — SIGMETRICS 2026, aceptado]
- **Huo, Leeds, Ludmir, DiBrita, Patel (Rice University) — Proceedings of ACM SIGMETRICS 2026 (arXiv:2510.06172).** "Anchor: Reducing Temporal and Spatial Output Performance Variability on Quantum Computers."
  Ejecutar el mismo programa en momentos o regiones distintas del chip produce resultados inconsistentes en QPUs NISQ; su técnica de programación lineal reduce la variabilidad un 73% de media. **Relevancia para el TFM:** complementa a Hirasaki — evidencia OPERACIONAL (impacto en outputs del usuario) frente a la evidencia FÍSICA (saltos en tasas de error). Citar ambos juntos al motivar `day_index` y el split temporal OOD en §3.

---

## Referencias — Paradigmas Alternativos de Mitigación

### QEM no supervisado — contraste de paradigma ✅ [REVISADO — IEEE QCE 2025]
- **Patil, Baron, Zhou — IEEE QCE 2025, pp. 849-860.** "Q-Cluster: Quantum Error Mitigation Through Noise-Aware Unsupervised Learning." DOI: 10.1109/QCE65121.2025.00097
  Clustering de bitstrings ruidosos por distancia Hamming → centroides via mayoría qubit-a-qubit → ajuste de distribución por inferencia Bayesiana. ExtraTrees regressor para estimar tasas de error efectivas desde calibración + features del circuito. 1.46× mejora de fidelidad media sobre output sin mitigar. Validado en 5 QPUs IBM. **Contraste con TFM:** paradigma no supervisado (sin Statevector ground truth) vs. nuestro supervisado. La ausencia de datos etiquetados es ventaja operacional pero limita la capacidad de corrección para ruido no-lineal complejo. Útil para §3 SOTA como comparativa de paradigmas.
