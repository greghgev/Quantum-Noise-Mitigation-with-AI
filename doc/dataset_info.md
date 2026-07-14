# Datasheet: TFM-Quantum Dataset

Este documento detalla las especificaciones, decisiones arquitectónicas y justificaciones matemáticas detrás del conjunto de datos generado para el entrenamiento de los modelos GEM (Gate Error Mitigation) y REM (Readout Error Mitigation).

El objetivo de este dataset **no** es el descubrimiento algorítmico, sino la captura y modelado del **Data Corruption & Denoising** inducido por el hardware cuántico en la era NISQ (Noisy Intermediate-Scale Quantum).

---

## 1. Hardware Objetivo y Límites de Escalabilidad

Para simular un entorno realista, el dataset actúa como un Gemelo Digital de los procesadores cuánticos de la familia **IBM Heron r2 (156 qubits)** — backend de referencia `ibm_kingston`. El proyecto migró desde IBM Eagle (`ibm_kyiv`) en julio de 2026, tras el retiro completo de la familia Eagle por parte de IBM (último Eagle retirado: abril 2026).

* **Topología Base:** Heron r2, 156 qubits, puerta 2-qubit nativa **CZ** (Eagle usaba CX). [SUPOSICIÓN] La calibración sintética temporal usa una cadena lineal de 20 qubits hasta disponer de credenciales IBM para descargar el coupling map real.
* **Límite de Qubits por Sub-grafo:** Entre **5 y 15 qubits**.
* **Justificación de Ingeniería:** La generación del *Ground Truth* (etiquetas $Y$) requiere simular el vector de estado ideal ($2^n$ amplitudes complejas). Escalar más allá de 15 qubits dispararía el consumo de memoria RAM por encima de los 16 GB disponibles en la estación de trabajo local, provocando cuelgues por *Out of Memory* (OOM). Operar en sub-grafos contiguos garantiza la escalabilidad del pipeline manteniendo tiempos de generación viables.

---

## 2. Composición Algorítmica

Para evitar el *overfitting* algorítmico y asegurar que las redes neuronales aprenden la física pura del ruido, el dataset se divide en tres grupos:

### Entrenamiento (~80% random + ~20% estructurado)

* **~80% Circuitos Aleatorios (Random Clifford / Unitary):** Exploran uniformemente el espacio de Hilbert. Obligan al modelo GEM a aprender la degradación subyacente sin memorizar patrones de algoritmos específicos. **La proporción exacta se ablacionará** (70/30, 80/20, 90/10) durante la fase de entrenamiento — no hay precedente empírico fijo en la literatura. Se elige la que maximice el rendimiento en validación.
* **~20% Circuitos Estructurados (en entrenamiento):**
  * **HEA (Hardware-Efficient Ansatz / VQE):** Circuitos variacionales parametrizados (bloques Ry + CX construidos manualmente — `TwoLocal` está deprecated en Qiskit 2.x —, `reps=2`). Sustituye a Grover. Profundidad O(reps×n), benchmark estándar VQE [Bao et al., ICLR 2025; Liao et al., Nature MI 2024].
  * **Trotterized TFIM (Transverse-Field Ising Model):** Simulación de Hamiltoniano de Ising mediante Trotterización. Benchmark estándar de QEM-Bench [Bao et al., ICML 2025]. Permite comparación directa con QEMFormer y el SOTA reciente.

### Zero-shot (NUNCA en entrenamiento — solo en evaluación)

* **QAOA (Quantum Approximate Optimization Algorithm):** Circuitos anchos con entrelazamiento entre qubits vecinos del coupling map. Es el benchmark más citado en el SOTA (GTraQEM, QEM-Bench, Q-Cluster). Generalizar a QAOA sin haberlo entrenado es el resultado más valorado por revisores.
* **QFT (Quantum Fourier Transform):** Circuitos de alta profundidad, entrelazamiento denso determinista, alta entropía en la distribución de salida. **Contribución original del TFM:** ningún paper de ML-QEM evalúa QFT sistemáticamente. Si el GEM generaliza a QFT, es un resultado novedoso. Si no, es una limitación empírica documentada honestamente.

---

## 3. Profundidad del Circuito (Circuit Depth)

* **Rango:** Entre **5 y 50 capas (layers)** de compuertas.
* **Justificación:** Es el espectro realista de la computación cuántica actual. Una profundidad menor a 5 apenas presenta ruido. Una profundidad mayor a 50 resulta en la destrucción térmica casi total de la señal cuántica, imposibilitando la recuperación de información y provocando que el modelo de IA aprenda ruido estocástico puro.

---

## 4. Estrategia Temporal y MLOps (Concept Drift)

El hardware de IBM no se recalibra de forma gradual — las tasas de error cambian de forma **escalonada** (saltos bruscos que persisten minutos u horas), según evidencia empírica de Hirasaki et al. (Applied Physics Letters, 2023). El dataset simula este comportamiento con 2–3 saltos aleatorios de ±15–30% en T1/T2 y readout_error repartidos a lo largo de los 10 días. El modelo previo de decay lineal (`1.0 - day_index * 0.02`) está descartado por ser factualmente incorrecto.

* **Train (Días 1 a 5):** Aprendizaje base.
* **Validation (Día 6):** Ajuste de hiperparámetros.
* **Brecha Temporal (Día 7):** Sin datos. Fuerza a la red a no depender de datos contiguos.
* **Test Estricto / OOD (Días 8 a 10):** Evaluación *Out-Of-Distribution*. Si el modelo mitiga el error con la calibración del día 10 habiendo visto solo hasta el día 5, se considera apto para producción.

---

## 5. Feature Engineering y Ground Truth

### Ground Truth (La Variable $Y$)

> **Alcance vigente (jul-2026):** la etiqueta Δ es el **error TOTAL** del circuito
> (puertas + readout juntos) — con el REM fuera de alcance, la separación de ruidos
> ya no aporta y el dataset actual es correcto tal cual.

A diferencia de los datos de entrada, el objetivo ideal no se calcula mediante muestreo empírico (*shots*), lo que induciría error estadístico (*Shot Noise*). Se extrae analíticamente usando `qiskit.quantum_info.Statevector` para asegurar precisiones matemáticas perfectas al penalizar la función de pérdida (*Loss Function*).

### Vectores de Características (La Variable $X$) — 19 dimensiones

El pipeline no exporta archivos genéricos (como CSV o Parquet) para evitar cuellos de botella en la lectura de CPU durante el entrenamiento. Los grafos se almacenan nativamente como tensores `.pt` (PyTorch Geometric). Cada nodo (compuerta cuántica) es un vector de 19 dimensiones:

| # | Feature | Dims | Fuente | Estado |
|---|---|---|---|---|
| 1 | **Instrucción Lógica:** One-Hot del tipo de compuerta (`cz`, `id`, `rz`, `sx`, `x`, `measure`) | 6 | Estructura del circuito | ✅ Implementado |
| 2 | **Rotaciones:** Ángulos paramétricos $\theta, \phi, \lambda$ (padding 0.0 si no aplica) | 3 | Parámetros del circuito | ✅ Implementado |
| 3 | **Telemetría qubit control:** $T_1$, $T_2$, `readout_error` | 3 | `backend.properties()` IBM | ✅ Implementado |
| 4 | **Error de compuerta** (`gate_error`): tasa empírica de error medida ese día en el hardware | 1 | `backend.properties()` IBM | ✅ Implementado |
| 5 | **Duración del pulso** (`gate_length`): tiempo real de ejecución en segundos | 1 | `backend.properties()` IBM | ✅ Implementado |
| 6 | **Telemetría qubit target:** $T_1$, $T_2$, `readout_error` (zero-pad `[0,0,0]` para puertas de 1 qubit) | 3 | `backend.properties()` IBM | ✅ Implementado (zero-pad correcto) |
| 7 | **Día de calibración** (`day_index / 10.0`): señal explícita de Concept Drift temporal | 1 | Pipeline de generación | ✅ Implementado |
| 8 | **Posición en el DAG** (`pos / depth_total`): profundidad normalizada — estima decoherencia acumulada | 1 | Estructura del DAG | ✅ Implementado |

**Nota Heron (CZ):** la puerta CZ es simétrica — no existe distinción física control/target. Por convención del extractor, "control" = `qargs[0]` y "target" = `qargs[1]` (orden asignado por el transpilador).

**Nota sobre puertas de 1 qubit:** Para puertas como `rz`, `sx`, `x`, `id` — que operan sobre un único qubit — la posición del *qubit target* (feature #6) se rellena con `[0.0, 0.0, 0.0]` (zero-padding), no con una copia del qubit control. Esto señala explícitamente al modelo que no existe un segundo qubit en la operación.

**Features #4 y #5** son estándar en la literatura de ML-QEM: QuESt [Wang et al., 2022, ICCAD] y GTraQEM [Bao et al., 2025, ICLR] las incluyen como componentes esenciales del vector de nodo. Disponibles directamente vía `backend.properties().gate_error()` y `backend.properties().gate_length()`.

**Limitación conocida — Δ global:** El GEM predice un único Δ escalar para todo el circuito. GTraQEM predice un Δ por observable (⟨Z_i⟩, ⟨Z_iZ_j⟩, etc.), lo que es más preciso cuando distintos observables tienen sensibilidades de ruido distintas. La simplificación del TFM reduce la complejidad de implementación. Ver `IDEAS_FUTURAS.md` para la extensión posible.

---

## 6. Métricas de Evaluación

Se reportan todas las métricas relevantes para comparación directa con el SOTA.

### GEM — predicción de error en valores esperados

| Métrica | Qué mide |
|---|---|
| MAE | Error medio absoluto en ⟨O⟩ |
| RMSE | Error cuadrático medio (penaliza errores grandes más que MAE) |
| R² | Correlación entre Δ predicho y Δ real (1.0 = perfecto, 0 = inútil) |
| Mejora relativa | Cuánto mejor es el TFM vs. no mitigar: `|⟨O⟩_noisy − ⟨O⟩_ideal|` vs `|⟨O⟩_mit − ⟨O⟩_ideal|` |

### REM — corrección de distribuciones de probabilidad

| Métrica | Qué mide |
|---|---|
| Hellinger Fidelity (HF) | Parecido entre P_mit y P_ideal. Rango [0, 1]; 1 = idénticos |
| Total Variation Distance (TVD) | Diferencia absoluta total entre distribuciones. 0 = perfecto |
| Ratio de mejora | HF_mit / HF_noisy. >1 significa que el REM ayuda |

### Granularidad del reporte

Todas las métricas se reportan separadas por:
- **In-distribution:** circuitos del mismo tipo que el training (random, HEA, TFIM no vistos)
- **Zero-shot A:** QAOA (nunca en training)
- **Zero-shot B:** QFT (nunca en training — contribución original)
- **Por día:** métricas del test OOD día a día para visualizar degradación por Concept Drift
