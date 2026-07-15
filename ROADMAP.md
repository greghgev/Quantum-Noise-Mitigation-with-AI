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

## ✅ TAREA 2: Esqueleto MLOps — COMPLETADA

- [x] `.gitignore` bien configurado (`data/`, `models/*.pt`, `mlflow.db`, `.env`, `*.TODO`)
- [x] `dvc init` + remote local configurado en `~/TFM/dvc-remote/`
- [x] `configs/gem_config.yaml` + `configs/rem_config.yaml` creados
- [x] `scripts/` con 4 scripts: `generate.py`, `train_gem.py`, `train_rem.py`, `evaluate.py`
- [x] `pre-commit install` ejecutado — hooks `black` + `flake8` activos en cada commit
- [x] `environment.yml` actualizado (flake8, pre-commit, dvc)
- [x] Repo GitHub limpio creado y primer commit subido (40 ficheros, historial desde cero)

---

## ✅ TAREA 1: Bugs en `src/quantum_gen.py` — COMPLETADA

- [x] **Bug 1 — Drift step-like:** `_init_drift_schedule()` genera 2–3 saltos aleatorios de ±15–30% por qubit sobre 10 días (Hirasaki et al., APL 2023). Decay lineal eliminado.
- [x] **Bug 2 — Operador QAOA:** `generate_qaoa()` construido manualmente con suma de términos ZZ por arista del coupling map. `QAOAAnsatz` deprecated eliminado.
- [x] **Bug 3 — Generador TFIM:** `generate_tfim()` implementado — Trotterized TFIM con parámetros J, h aleatorios y pasos de Trotter configurables.
- [x] **Bug 4 — Split zero-shot:** `generate_dataset()` enruta por tipo de split. Train/val/test → [random, hea, tfim]. zeroshot_qaoa → [qaoa]. zeroshot_qft → [qft]. QAOA/QFT nunca aparecen en train.
- [x] **Bug 5a — 19 dims:** `circuit_to_graph()` produce vectores de 19 dims: +gate_error, +gate_length, +day_index, +depth_position.
- [x] **Bug 5b — Zero-pad:** puertas 1q tienen `vec_target = [0.0, 0.0, 0.0]` correctamente.
- [x] **Deprecated Qiskit 2.x:** `QFT` → `synth_qft_full`, `TwoLocal` → HEA manual Ry+CX, `QAOAAnsatz` → QAOA manual.
- [x] Suite de tests: **36/36 PASSED** — `tests/test_quantum_gen.py`
- [x] Compatibilidad Qiskit: `qiskit-aer` actualizado a 0.17.2 (compatible con qiskit 2.4.1). `environment.yml` actualizado.

---

## ✅ TAREA 3: Generación del dataset (mini-run primero) — COMPLETADA

- [x] Calibración sintética `ibm_kyiv_calib.json` generada (20 qubits, cadena lineal, valores realistas Eagle r3)
      Nota: credenciales IBM no conectaron (CRN no coincide con token actual) — para el full run usar calibración real
- [x] Añadir `run()` a `TFMDatasetPipeline` — método faltante que bloqueaba `scripts/generate.py`
- [x] Implementar `build_noise_model_for_day()` — modelo de ruido por día con drift (depolarización + readout)
      Nota: thermal_relaxation_error + depolarizing_error juntos producen Kraus vacíos en Aer 0.17 — solo depolarización por estabilidad
- [x] Añadir simulación ruidosa Aer en `generate_dataset()` — compute Δ = ⟨Z⟩_ideal − ⟨Z⟩_noisy
- [x] Actualizar `circuit_to_graph()` — `y = Δ` (GEM target escalar) + `ideal_probs` + `noisy_probs`
- [x] Generar mini-dataset: 178 muestras totales (104 train + 22 val + 22 test + 15 QAOA + 15 QFT), 2 días
- [x] `dvc add data/processed/ data/raw/` + `dvc push` — 181 archivos versionados en remote
- [ ] Generar dataset completo (split temporal completo: días 1–10) — pendiente del visto bueno del usuario + credenciales IBM
      ⚠️  Rango de qubits 5–15 es correcto para Galaxy Book 16 GB. Posible mejora: 5–18 para runs nocturnos (anotado en TAREA 7)

---

## ✅ TAREA 3b: Migración Eagle → Heron — COMPLETADA (jul 2026)

**Motivo:** ibm_kyiv retirado por IBM el 18-abr-2025. Toda la familia Eagle retirada desde abr-2026
(últimos: ibm_brussels, ibm_strasbourg — 27-abr-2026). No existe alternativa Eagle con calibración real.

- [x] Backend objetivo: `ibm_kingston` (Heron r2, 156 qubits, Open Plan) — configurable vía `IBM_BACKEND_NAME` en `.env`
- [x] `src/config.py`: nuevas constantes `BACKEND_NAME` y `BASIS_GATES = [cz, id, rz, sx, x]`
- [x] `src/quantum_gen.py`: basis gates CZ, one-hot posición 0 = `cz`, noise model 2q sobre `cz` en ambas orientaciones del par
- [x] `scripts/make_synthetic_calib.py`: genera calibración sintética Heron r2 [SUPOSICIÓN: valores plausibles, cadena lineal 20q] (script eliminado en jul-2026 al obtener calibración real — historial git)
- [x] `tests/test_quantum_gen.py` actualizado (cx→cz) — 36/36 PASSED
- [x] Mini-dataset Eagle borrado y regenerado con Heron; docs actualizadas (CLAUDE, README, dataset_info, IDEA_GENERAL...)
- [ ] Credenciales IBM: cuenta antigua expirada (0 instancias); cuenta nueva bloqueada por verificación de tarjeta
      → email enviado a verify@us.ibm.com (código REG-PAYGO-UPGRADE-16W8VLK45NPC1). Al verificar:
      crear instancia Open Plan us-east, poner CRN+token en .env, borrar ibm_kingston_calib.json y regenerar dataset con calibración real

---

## ✅ TAREA 8: Fixes de calidad de datos detectados en EDA — COMPLETADA (jul 2026)

Hallazgos del EDA post-migración Heron, verificados empíricamente (no solo sospechados).
Ver `doc/migracion_heron.md`.

- [x] **Fix 1 — Leakage train/val/test en modo mini:** `sample_seed = SEED + i + day*10000`
      no incluía el split → mismo día + mismo índice = circuito IDÉNTICO en train/val/test
      (verificado: `torch.equal(train.x, val.x) == True`). Fix: `_split_seed_offset(split_name)`
      con crc32 (estable entre procesos, a diferencia de `hash()`). Test de regresión añadido:
      `test_no_leakage_between_splits_same_day`.
- [x] **Fix 3 — `shots=4096` hardcodeado:** movido a `config.py` como `LABEL_SHOTS`, distinto
      de `TRAIN_SHOTS` (histogramas realistas, uso futuro). Documentada la diferencia en `fallas_y_soluciones.md`
      (que tenía la nota de shots desactualizada — mezclaba ambos usos).
- [x] **Fix 2 (parcial) — QAOA con `reps` variable 1-3** en lugar de fijo en 1: más profundidad
      → más error real acumulado, mejor relación señal/ruido sin subir shots.
- [x] `doc/fallas_y_soluciones.md` §3 corregido: la nota de "Equipo B genera QAOA, QFT" para
      entrenar el GEM contradecía el diseño zero-shot vigente desde TAREA 1 (QAOA/QFT NUNCA
      en training). Corregido a Random/HEA/TFIM.
- [ ] **Pendiente — subir `LABEL_SHOTS` a 8192 o 16384 antes del full run.** [SUPOSICIÓN] Con
      4096 shots el suelo de shot-noise (±0.005) es del mismo orden que Δ en QAOA/QFT poco
      profundos — la etiqueta zero-shot es en esos casos indistinguible del ruido de medición.
      No se sube en los mini-runs por coste de cómputo; decisión a tomar ANTES del full run
      (la calibración real ya está disponible desde jul-2026).
- [x] Mini-dataset regenerado con los 3 fixes; notebook 01 re-ejecutado y verificado
      `val != train` (antes eran idénticos).

---

## 🔄 CAMBIO DE ALCANCE (jul-2026, acordado con el tutor)

**Decisión**: el TFM se reduce a **solo el módulo GEM**, planteado como **comparativa de modelos**.
La corrección de lectura post-ejecución del diseño original pasa a trabajo futuro (diseño completo consolidado en `IDEAS_FUTURAS.md` IDEA-0 — única referencia en el repo). Razones del tutor: alcance
original excesivo para el plazo; el GEM solo ya es contribución suficiente (idea poco estudiada +
arquitecturas novedosas + dataset propio + evaluación comparativa).

**Consecuencias técnicas:**
1. **Comparativa de 3 modelos** prediciendo Δ: Ridge (regresión lineal clásica) + Random Forest
   (mejor baseline según Liao 2024) + Graph Transformer (GEM). Baselines tabulares sobre features
   agregadas del circuito; mismo protocolo de evaluación para los tres.
2. **Δ pasa a ser el error TOTAL (puertas + readout)** — sin el segundo módulo, la separación de
   ruidos ya no aporta. ✅ El dataset actual ya es correcto tal cual: el desacoplamiento pendiente queda ANULADO.
3. **Drift real**: cuando la cuenta IBM esté activa, recopilar la calibración real diaria (gratis,
   sin QPU) y usar los días nuevos como test temporal — sustituye al drift sintético [SUPOSICIÓN]
   como conjunto de validación (idea del tutor, punto 3 de su correo).

---

## ⏳ TAREA 4: Implementación de modelos — comparativa GEM vs. baselines

> 🔒 Espera a TAREA 3. Alcance revisado jul-2026 (ver banner de arriba).

- ~~Desacoplar el ruido en la generación de datos~~ **ANULADO** — con el alcance GEM-only, Δ es el
  error total y el dataset actual ya es correcto (ver Cambio de Alcance)
- [ ] Proporciones random/estructurados configurables en `generate_dataset()` (ahora hardcodeadas
      80/10/10) — necesario para la ablación 70/30–90/10 del notebook 05 (opcional si falta tiempo)
- [ ] `src/dataset.py` — PyG Dataset + DataLoaders (carga separada in-distribution / zero-shot)
- [ ] `src/features.py` — **features agregadas para baselines tabulares**: conteos por tipo de
      puerta, nº qubits, profundidad, telemetría media/mín/máx de los qubits usados, day_index
      (estilo global-features de Liao 2024/QEMFormer, citable)
- [ ] `src/baselines.py` — Ridge + Random Forest (sklearn) con búsqueda ligera de hiperparámetros
- [ ] `src/utils.py` — métricas GEM: MAE, RMSE, R², mejora relativa (idénticas para los 3 modelos)
- [ ] `src/gem_model.py` — Graph Transformer con:
  - Nodo virtual QCR (Quantum Circuit Representative) conectado a todos los nodos
  - No message-passing (estilo GTraQEM, no MPNN)
  - MLP de salida que lee el embedding del QCR para predecir Δ
- [ ] `src/train.py` — entrenamiento de los 3 modelos con protocolo IDÉNTICO (mismos splits, mismas
      métricas, misma semilla) + MLflow + normalización de Δ
- [ ] `configs/baselines_config.yaml` — hiperparámetros de Ridge y RF (sin hardcoding)

---

## ⏳ TAREA 5: Notebooks de análisis y evaluación

> 🔒 Espera a TAREA 4.

> Plan revisado (jul-2026) tras completar el 01 y detectar el hueco de desacoplamiento (ver TAREA 4).

- [x] `01_eda_ibm_telemetry.ipynb` — ✅ COMPLETADO con rol ampliado: EDA divulgativa (circuito
      dibujado, DAG coloreado, calibración, heatmap de drift, dataset con franja de shot-noise,
      conclusiones sobre outputs reales). Sirve también como pieza de comunicación para el tutor.
- [ ] `02_quantum_circuit_prototyping.ipynb` — REENFOCADO para no duplicar el 01: validación de los
      5 tipos de circuito lado a lado + **barrido de shots (1024→16384) midiendo la estabilidad de
      la etiqueta Δ** → justificación empírica de la decisión LABEL_SHOTS (responde al suelo de
      shot-noise detectado en TAREA 8)
- [ ] `03_gem_transformer_evaluation.ipynb` — entrenamiento y métricas del GEM (MAE, RMSE, R²,
      mejora relativa) separadas por: in-distribution / zero-shot QAOA / zero-shot QFT / por día
      (la nota anterior sobre regenerar etiquetas sin readout queda ANULADA — Δ es error total)
- [ ] `04_model_comparison.ipynb` — **LA COMPARATIVA (núcleo del TFM revisado)**: Ridge vs. Random
      Forest vs. GEM con protocolo idéntico, tabla de métricas por split, análisis de dónde gana
      cada modelo.
- [ ] `05_final_pipeline_tradeoffs.ipynb` — análisis final: mejor modelo aplicado al flujo completo,
      ablación 70/30 vs 80/20 vs 90/10 (opcional), figuras/tablas exportadas para la memoria.
      Trade-offs con barras agrupadas o small multiples en lugar de radar (legibilidad)

---

## ⏳ TAREA 6: Datos reales de IBM (calibración diaria + validación en hardware)

> Requiere cuenta IBM verificada (pendiente: email a verify@us.ibm.com). La recopilación de
> calibración es GRATIS (no gasta QPU) — arrancarla EN CUANTO haya cuenta, cada día suma.

### 6a — Recopilación de calibración real diaria (idea del tutor — test con drift auténtico)
- [x] ✅ Cuenta IBM ACTIVA (14-jul-2026): upgrade aprobado, instancia Open Plan us-east, credenciales verificadas
- [x] Script `scripts/collect_calibration.py` creado y probado (descarga y archiva
      `data/raw/calib_history/ibm_kingston_YYYY-MM-DD.json`)
- [x] ✅ VERIFICADO: la API SÍ permite calibraciones históricas (`backend.properties(datetime=...)`) —
      `--backfill 30` descargó 29 días reales de golpe (jun–jul 2026), sin espera
- [x] Calibración real descargada; dataset regenerado con ella (fix EdgeList aplicado); sintética eliminada
- [ ] Seguir ejecutando el recolector a diario (manual o cron) para alargar el histórico
- [ ] Regenerar dataset con el HISTÓRICO real por día: train = días antiguos, test = días recientes
      (drift REAL sustituye al drift sintético step-like) — requiere adaptar HardwareTelemetrics
      para leer snapshots de calib_history/ en lugar del drift simulado

### 6b — Validación en hardware real (brecha simulation-to-real)
- [ ] Ejecutar 20–30 circuitos representativos en `ibm_kingston` real (incluir random, HEA, QAOA, QFT)
- [ ] Comparar MAE_real vs. MAE_simulado para verificar la brecha simulation-to-real
- [ ] Si la brecha es aceptable (<20% de degradación): documentar como validación exitosa
- [ ] Si la brecha es grande: documentar como limitación honesta en §6 del TFM y proponer fine-tuning como trabajo futuro (ver `IDEAS_FUTURAS.md` IDEA-4)

---

## ⏳ TAREA 7: Mejoras al modelo de ruido (post-validación real)

> 🔒 Espera a TAREA 6. Activar solo si la brecha simulation-to-real justifica el esfuerzo.
> Prioridad REBAJADA tras el cambio de alcance (jul-2026): estas mejoras servían sobre todo al
> módulo de lectura descartado (IDEAS_FUTURAS IDEA-0). Para el GEM solo importan si la brecha de TAREA 6b es grande.

### 7a — Matriz de confusión de readout asimétrica

**Motivación**: IBM expone dos probabilidades independientes por qubit — `P(leer 1 | estado 0)` y `P(leer 0 | estado 1)`. El código actual usa un único escalar `readout_error` que asume simetría. En hardware real la asimetría es significativa (típicamente P(1|0) ≫ P(0|1)).

**Impacto en código**: moderado, con cambio en cascada en la especificación de features.

- [ ] Verificar qué campo expone `backend.properties()` en `qiskit-ibm-runtime >= 0.40` para P(1→0) por separado (campo esperado: `prob_meas1_prep0`, pero requiere confirmación contra la API real)
- [ ] Actualizar `_fetch_from_ibm_and_save()`: guardar `readout_p01` y `readout_p10` por qubit en el JSON de calibración
- [ ] Actualizar `HardwareTelemetrics.get_qubit_features()`: devolver ambas probabilidades
- [ ] Actualizar `circuit_to_graph()`: vector de features **pasa de 19 → 20 dims** (añadir `readout_p10` en posición 15)
  - Nueva distribución: `[0:6 one-hot][6:9 params][9:12 ctrl T1/T2/re][12 gate_error][13 gate_length][14:17 target T1/T2/re][17 readout_p10][18 day_norm][19 depth]`
- [ ] Actualizar `tests/test_quantum_gen.py`: cambiar todos los asserts `== 19` a `== 20`
- [ ] Actualizar `CLAUDE.md` §4, `doc/dataset_info.md` y `README.md` con nueva especificación

### 7b — Crosstalk entre qubits vecinos (solo si la brecha es > 20%)

**Motivación**: `NoiseModel.from_backend()` no modela el acoplamiento ZZ residual entre qubits físicamente cercanos. Si la brecha simulation-to-real es grande, el crosstalk es un candidato probable.

**Decisión de diseño**: NO añadir ahora. Razones:
1. Las tasas de acoplamiento ZZ reales no están expuestas de forma consistente en `backend.properties()` para `ibm_kingston` — cualquier valor sería una suposición especulativa.
2. La versión básica (error de despolarización extra uniforme para todas las puertas CX simultáneas) introduce ruido no calibrado que puede empeorar el modelo.
3. El diagnóstico correcto viene de los datos de TAREA 6: si la brecha es pequeña, el crosstalk no importa.

**Plan condicional** (ejecutar solo si se activa este bloque):
- [ ] Buscar en `backend.target` o en los calibration payloads de IBM si hay tasas ZZ disponibles para `ibm_kingston` en el momento de la validación
- [ ] Si hay datos reales: añadir `depolarizing_error` extra calibrado por par de qubits en `_build_noise_model()`
- [ ] Si no hay datos reales: documentar como limitación conocida en §6 del TFM — NO inventar la magnitud
