# ROADMAP — TFM-Quantum

> Leer junto a CLAUDE.md al inicio de cada sesión.
> Formato: ✅ Completado | 🔄 En curso | ⏳ Pendiente | 🔒 Bloqueado por tarea anterior

---

## ✅ Completadas

- Estructura base del proyecto (`src/`, `data/`, `doc/`, `notebooks/`)
- `src/config.py` — rutas, constantes físicas, semillas
- `src/quantum_gen.py` — CircuitFactory, HardwareTelemetrics, QuantumGraphExtractor, TFMDatasetPipeline (implementación base)
- SOTA completa (2 PDFs en `doc/SOTA/`) + análisis comparativo en `doc/SOTA/comparative_analysis.md`
- Tests eliminados para empezar desde cero (TEST_exhaustivo.py, TEST_ibm_minimal.py) — pendiente crear suite nueva en TAREA 1
- Sustitución Grover → HEA en CircuitFactory y pipeline
- Inyección de dependencia en TFMDatasetPipeline
- Documentación actualizada (CLAUDE.md, dataset_info.md)

---

## 🔄 En curso — TAREA 2: Esqueleto MLOps

Crear la estructura de organización MLOps antes de generar datos ni implementar modelos.

### Pasos pendientes dentro de TAREA 2:
- [ ] `git init` + `.gitignore` bien configurado (excluir `data/`, `models/`, `*.pt`, `mlflow.db`, `.env`)
- [ ] `dvc init` + configurar remote local (carpeta o disco externo)
- [ ] Crear `configs/` con YAMLs stub para experimentos GEM y REM
- [ ] Crear `scripts/` con los 4 scripts de ejecución (generate, train_gem, train_rem, evaluate)
- [ ] Configurar pre-commit hooks (`black` + `flake8`)

---

## ⏳ TAREA 1: Arreglar bugs en `src/quantum_gen.py`

> 🔒 Espera a que TAREA 2 esté completa.

- [ ] **Bug 1 — Drift step-like en `build_noise_model_for_day()`:** implementar 2–3 saltos aleatorios de ±15–30% en T1/T2/readout_error a lo largo del timeline de 10 días. El decay lineal previo está descartado (Hirasaki et al., APL 2023).
- [ ] **Bug 2 — Operador QAOA trivial:** reemplazar `SparsePauliOp("Z" * n)` por suma de términos `ZZ` sobre las aristas del coupling map.
- [ ] **Bug 5 — Vector de features 15 → 19 dims:** añadir `gate_error`, `gate_length`, `day_index`, `depth_position`; corregir zero-pad para puertas 1q (actualmente copia el qubit control en lugar de zeros).
- [ ] **Bug nuevo — Generador Trotterized TFIM:** implementar generación de circuitos TFIM (Transverse-Field Ising Model) en `CircuitFactory`. Benchmark estándar de QEM-Bench para comparación con el SOTA.
- [ ] **Bug nuevo — Split zero-shot:** modificar `TFMDatasetPipeline` para que QAOA y QFT nunca aparezcan en el training set. Guardar en carpetas separadas: `data/processed/train/`, `data/processed/zeroshot_qaoa/`, `data/processed/zeroshot_qft/`.
- [ ] Crear nueva suite de tests desde cero: validar 19 dims, drift step-like, tipos de circuito (random/HEA/TFIM/QAOA/QFT), split zero-shot.

---

## ⏳ TAREA 3: Generación del dataset (mini-run primero)

> 🔒 Espera a TAREA 1.

- [ ] Verificar conexión IBM y descarga de `ibm_kyiv_calib.json`
- [ ] Generar dataset pequeño (100–200 muestras, días 0–2) para validar pipeline end-to-end
- [ ] `dvc add data/processed/` — versionar primera generación
- [ ] Generar dataset completo (split temporal completo: días 1–10)

---

## ⏳ TAREA 4: Implementación de modelos

> 🔒 Espera a TAREA 3.

- [ ] `src/dataset.py` — PyG Dataset + DataLoaders (incluir carga separada de in-distribution y zero-shot)
- [ ] `src/utils.py` — implementar TODAS las métricas:
  - GEM: MAE, RMSE, R², mejora relativa
  - REM: Hellinger Fidelity, Total Variation Distance, ratio de mejora HF_mit/HF_noisy
  - GMRES wrapper, One-Hot encodings
- [ ] `src/gem_model.py` — Graph Transformer con:
  - Nodo virtual QCR (Quantum Circuit Representative) conectado a todos los nodos
  - No message-passing (estilo GTraQEM, no MPNN)
  - MLP de salida que lee el embedding del QCR para predecir Δ
- [ ] `src/rem_model.py` — GNN local 2×2 + Matrix-Free GMRES
- [ ] `src/train.py` — bucles desacoplados GEM/REM + MLflow + normalización de Δ por rango del observable
- [ ] `src/inference.py` — pipeline producción GEM → QPU → REM

---

## ⏳ TAREA 5: Notebooks de análisis y evaluación

> 🔒 Espera a TAREA 4.

- [ ] `01_eda_ibm_telemetry.ipynb` — análisis exploratorio + visualización del drift step-like simulado
- [ ] `02_quantum_circuit_prototyping.ipynb` — validación de los 5 tipos de circuito (random, HEA, TFIM, QAOA zero-shot, QFT zero-shot)
- [ ] `03_gem_transformer_evaluation.ipynb` — métricas GEM (MAE, RMSE, R², mejora relativa) separadas por: in-distribution / zero-shot QAOA / zero-shot QFT / por día
- [ ] `04_rem_gnn_matrix_free.ipynb` — métricas REM (HF, TVD, ratio de mejora) con los mismos splits
- [ ] `05_final_pipeline_tradeoffs.ipynb` — pipeline completo + ablación 70/30 vs 80/20 vs 90/10

---

## ⏳ TAREA 6: Validación en hardware IBM real

> 🔒 Espera a TAREA 5. Requiere créditos IBM activos.

- [ ] Ejecutar 20–30 circuitos representativos en `ibm_kyiv` real (incluir random, HEA, QAOA, QFT)
- [ ] Comparar MAE_real vs. MAE_simulado para verificar la brecha simulation-to-real
- [ ] Si la brecha es aceptable (<20% de degradación): documentar como validación exitosa
- [ ] Si la brecha es grande: documentar como limitación honesta en §6 del TFM y proponer fine-tuning como trabajo futuro (ver `IDEAS_FUTURAS.md` IDEA-4)
