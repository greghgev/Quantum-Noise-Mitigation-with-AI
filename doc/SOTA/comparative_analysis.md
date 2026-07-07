# Análisis Comparativo SOTA — TFM-Quantum
## Comparativa por Categoría de Decisión Arquitectónica

> **Convención:** `[HECHO]` = resultado empírico verificable de un paper peer-reviewed o preprint.
> `[SUPOSICIÓN TFM]` = decisión de diseño del TFM, no validada empíricamente hasta que haya datos reales.
> Los papers marcados ⚠️ son preprints sin peer review confirmado.

---

## 0. Confirmación de PDFs Recibidos

| Paper | Estado en §8 | PDF |
|---|---|---|
| GTraQEM — Bao et al. ICLR 2025 | ✅ peer-reviewed | ✅ Recibido |
| QEMFormer — Bao et al. ICML 2025 | ✅ peer-reviewed | ✅ Recibido |
| Liao et al. 2024 — Nature MI | ✅ peer-reviewed | ✅ Recibido |
| Liao et al. 2025 — npj QI | ✅ peer-reviewed | ✅ Recibido |
| Liu et al. — Frontiers of Physics 2026 | ✅ peer-reviewed | ✅ Recibido |
| M3 / Nation et al. — PRX Quantum 2021 | ✅ peer-reviewed | ✅ Recibido |
| Kim et al. — NJP 2022 | ✅ peer-reviewed | ✅ Recibido |
| Lee & Park — ML:ST 2023 | ✅ peer-reviewed | ✅ Recibido |
| Hirasaki et al. — APL 2023 | ✅ peer-reviewed | ✅ Recibido |
| Q-Cluster — Patil et al. IEEE QCE 2025 | ✅ peer-reviewed | ✅ Recibido |
| Placidi et al. — arXiv:2601.14226 | ⚠️ preprint | ✅ Recibido |
| Das et al. — arXiv:2512.14541 | ⚠️ preprint | ✅ Recibido |
| Korolev et al. — arXiv:2606.02697 | ⚠️ preprint (EXTRA, añadido a §8) | ✅ Recibido |

**Total: 13 PDFs.** Todos los papers de §8 están presentes. Korolev et al. añadido a §8 por petición explícita.

---

## 1. Representación del Circuito como Grafo

### Lo que hacen los papers

| Paper | Representación | Tipo de arista | Notas |
|---|---|---|---|
| GTraQEM ✅ | DAG — nodos=puertas, aristas=dependencias de qubit | Dirigida qubit→qubit | No-message-passing GT |
| QEMFormer ✅ | DAG (rama secundaria) + P_noisy global (rama primaria) | Dirigida | El grafo aporta contexto; P_noisy domina |
| Liao 2024 ✅ | GNN message-passing sobre el DAG del circuito | No especificada en detalle | MPNN-style |
| Liu et al. ✅ | DAG del circuito | Dirigida por qubit | GNN supera a CNN a mayor profundidad |
| Das et al. ⚠️ | DAG del circuito con features topológicas (betweenness centrality, k-core) | Dirigida | Objetivo: forense, no mitigación |
| M3 ✅ | No usa representación de circuito | — | Opera post-ejecución sobre bitstrings |
| Kim et al. ✅ | No usa representación de circuito | — | Opera sobre mediciones de calibración |
| Lee & Park ✅ | No usa representación de circuito | — | Opera sobre mediciones de calibración |
| Q-Cluster ✅ | No usa representación de circuito | — | Features escalares del circuito transpilado |
| Korolev et al. ⚠️ | Features escalares del circuito (parámetros variacionales) | — | Específico VQE/QAOA |
| Hirasaki ✅ / Liao 2025 ✅ / Placidi ⚠️ | No centrado en representación de grafo | — | Contextos distintos |

**[HECHO] GTraQEM:** El Graph Transformer *sin* message-passing supera al GNN con message-passing estándar en tareas de mitigación de expectation values. Justificación: en un DAG cuántico, la información de qubit fluye en una sola dirección; el message-passing bidireccional introduce ruido agregado innecesario.

**[HECHO] Liu et al.:** La GNN supera a la CNN en extrapolación a circuitos más profundos que los del conjunto de entrenamiento (entrenado en 3-7 qubits, testeado en 16 qubits, R²≥0.92).

**[HECHO] Das et al.:** Features de topología del grafo (betweenness centrality, k-core) sobre el DAG logran Spearman ρ=0.98 para inferir tasas de error sin calibración explícita.

### TFM

**[SUPOSICIÓN TFM]** DAG con nodos=puertas y aristas dirigidas por dependencia de qubit. La elección de no-message-passing vs. message-passing aún no está fijada en el código.

**Análisis de trade-off:**
- La elección de DAG está bien respaldada (GTraQEM, Liu et al., Das et al. convergen en este punto).
- La elección entre no-message-passing (GTraQEM) vs. message-passing (Liao 2024, Liu et al.) es la decisión arquitectónica más incierta. El SOTA más fuerte (GTraQEM, QEMFormer) usa no-message-passing → el TFM debería adoptar no-message-passing como decisión inicial y ablacionar después.
- El nodo virtual QCR de GTraQEM (conectado a todos los nodos) añade contexto global sin message-passing — es un truco elegante que el TFM no tiene actualmente.

---

## 2. Vector de Features de Nodo

### Lo que hacen los papers

**GTraQEM ✅ (ablación implícita en el paper):**
- One-hot gate type (basis gates IBM: cx, id, rz, sx, x)
- Ángulos paramétricos θ, φ, λ
- T1, T2, readout_error del qubit control
- T1, T2, readout_error del qubit target (zero-pad para puertas 1q)
- gate_error (tasa empírica del día)
- gate_length (duración del pulso)
- **Total: ~14-16 dims** (no reportado explícitamente en el paper)

**QEMFormer ✅:**
- Gate type one-hot + multi-hot qubit encoding + ángulos (por nodo)
- Features globales adicionales: 8 stats del circuito + expectation values Pauli (3×N_q) + P_noisy
- El multi-hot qubit encoding NO es generalizable a N arbitrario (encoding fijo por número de qubits)

**Placidi et al. ⚠️ — ABLATION STUDY CLAVE:**
> [HECHO] "P_noisy (el valor esperado ruidoso ya medido) es el feature dominante. T1/T2/gate_error/readout_error son features secundarios que aportan mejora marginal pero consistente. Sin hardware features, el modelo degrada moderadamente; sin P_noisy, el modelo colapsa." — resultado empírico en hardware Quantinuum real.

**Q-Cluster ✅ — FEATURE IMPORTANCE EMPÍRICO (ExtraTrees):**
> [HECHO] La importancia de features para predecir la tasa de error efectiva, en orden descendente: ESP (Estimated Success Probability = función de gate_error + readout_error + 2q_error) > #qubits medidos > #qubits totales > entropía del output ruidoso > #2q gates > #basis gates (SX, X, RZ).

**Liu et al. ✅:** Vector de 31 dims incluyendo: one-hot de tipo de puerta (más tipos que IBM basis set), coordenadas de qubit en la topología, T1/T2, gate_error, profundidad relativa en el DAG, número de vecinos en el grafo.

**Liao 2024 ✅:** T1/T2/gate_error como features principales. Demuestra que la telemetría de hardware es crítica para la generalización entre días de calibración.

**Kim et al. ✅:** No usa features de nodo de grafo — opera sobre mediciones directas. Relevante para REM.

### TFM

**[SUPOSICIÓN TFM]** Vector de 19 dims:
| # | Feature | Dims | Estado |
|---|---|---|---|
| 1 | one-hot gate type (cx, id, rz, sx, x, measure) | 6 | ✅ implementado |
| 2 | ángulos θ, φ, λ | 3 | ✅ implementado |
| 3 | T1, T2, readout_error qubit control | 3 | ✅ implementado |
| 4 | gate_error | 1 | ❌ pendiente Bug 1 |
| 5 | gate_length | 1 | ❌ pendiente Bug 1 |
| 6 | T1, T2, readout_error qubit target (zero-pad para 1q) | 3 | ⚠️ bug: copia en lugar de zero-pad |
| 7 | day_index / 10.0 | 1 | ❌ pendiente Bug 1 |
| 8 | depth_position / depth_total | 1 | ❌ pendiente Bug 1 |

**Análisis de trade-off:**
- Las 19 dims son competitivas con GTraQEM en cobertura de información. El diseño es sólido.
- `gate_error` y `gate_length` están empíricamente justificados por GTraQEM y Q-Cluster (ESP los incluye).
- **Problema crítico:** El TFM NO incluye P_noisy como feature — es un modelo pre-ejecución. Placidi demuestra que P_noisy domina en el paradigma post-ejecución. El TFM debería explicitar en §3/§4 que opera en un paradigma diferente (no porque P_noisy no sea importante, sino porque el GEM predice antes de ejecutar).
- `day_index` como feature explícito del drift NO aparece en ningún paper revisado — es una contribución original del TFM. Bien justificado teóricamente por Hirasaki et al. pero sin validación empírica en los papers.
- El multi-hot qubit encoding de QEMFormer no es generalizable — el diseño de 19 dims del TFM es más robusto en este sentido.
- **Bug crítico activo:** feature #6 (target qubit zero-pad) copia el qubit control en lugar de rellenar con ceros → introduce información falsa para puertas de 1 qubit.

---

## 3. Arquitectura del Modelo Principal (GEM)

### Lo que hacen los papers

**GTraQEM ✅ — Referencia arquitectónica directa:**
- Graph Transformer sin message-passing
- Nodo virtual QCR (Quantum Circuit Representative): se conecta a todos los nodos del grafo; actúa como receptor de información global sin contaminar los embeddings locales con aggregation.
- Atención por aristas (edge-aware attention)
- Output: escalar Δ por observable (ZZ, Z, etc.)
- [HECHO] Supera a GNN con message-passing en todos los benchmarks de expectation value mitigation.

**QEMFormer ✅ — Nuevo SOTA:**
- Dos ramas: (a) MLP sobre metadata global del circuito, (b) Graph Transformer sobre la estructura del DAG
- Las ramas se combinan con P_noisy mediante mecanismo de atención
- [HECHO] QEMFormer > GTraQEM en todos los settings de QEM-Bench (ibm_kyiv: MAE=0.018, RMSE=0.026)
- La mejora sobre GTraQEM se atribuye principalmente a la adición de P_noisy como contexto global.

**Liao 2024 ✅:**
- GNN message-passing (estilo MPNN) + MLP sobre features de hardware
- [HECHO] Escala hasta 100 qubits en hardware IBM real sin degradación significativa

**Liu et al. ✅:**
- GNN sobre DAG; testada contra CNN para misma tarea
- [HECHO] GNN > CNN para extrapolación de profundidad y número de qubits
- [HECHO] Baseline tabular (gradient boosting) es competitivo para circuitos pequeños (<10 qubits)

**Korolev et al. ⚠️:**
- ML ligero (no GNN) sobre parámetros variacionales + telemetría
- Solo para VQE/QAOA; no generalizable a circuitos arbitrarios
- [HECHO — preprint sin revisión] Near-Clifford training generaliza a VQE con parámetros lejanos de 0/π.

### TFM

**[SUPOSICIÓN TFM]** Graph Transformer (similar a GTraQEM). Sin nodo virtual QCR actualmente especificado. Sin la segunda rama MLP de QEMFormer.

**Análisis de trade-off:**
- La arquitectura base (Graph Transformer) está bien respaldada por el SOTA (GTraQEM, QEMFormer).
- La ausencia del nodo virtual QCR es una omisión respecto a GTraQEM — es el componente que permite contexto global sin message-passing. **Recomendación fuerte: añadirlo.**
- No incluir la rama MLP de QEMFormer ni P_noisy es consecuencia del paradigma pre-ejecución del TFM — es una diferenciación válida, no una omisión.
- El TFM no especifica el número de capas de atención ni las dimensiones del embedding — estas son decisiones de hiperparámetros que aún no están fijadas.

---

## 4. Target Variable y Formulación del Output

### Lo que hacen los papers

| Paper | Target | Paradigma |
|---|---|---|
| GTraQEM ✅ | Δ = ⟨O⟩_noisy − ⟨O⟩_ideal (por observable) | Pre-ejecución parcial |
| QEMFormer ✅ | ⟨O⟩_mit directamente (desde P_noisy) | Post-ejecución |
| Liao 2024 ✅ | ⟨O⟩ post-mitigado | Post-ejecución |
| Liao 2025 ✅ | ⟨O⟩ post-mitigado (sin ground truth de Statevector) | Post-ejecución + data augmentation |
| Liu et al. ✅ | Distribución completa P_ideal | Post-ejecución |
| M3 ✅ | P_ideal redistribuida desde P_noisy | Post-ejecución |
| Kim et al. ✅ | Distribución de medición corregida por qubit | Post-ejecución (calibración) |
| Lee & Park ✅ | Distribución corregida en subespacio de bitstrings | Post-ejecución |
| Q-Cluster ✅ | P_ideal redistribuida (sin ML para el target) | Post-ejecución |
| Placidi ⚠️ | ⟨O⟩_mit desde P_noisy | Post-ejecución |
| Korolev ⚠️ | ⟨O⟩_mit para VQE | Post-ejecución (parámetros variacionales) |

**[HECHO] GTraQEM:** Predice Δ = ⟨O⟩_noisy − ⟨O⟩_ideal, no el valor absoluto. Esta formulación residual estabiliza el entrenamiento porque Δ es numéricamente pequeño comparado con ⟨O⟩.

**[HECHO — implicación del paradigma]:** Todos los papers que predican directamente ⟨O⟩_mit (QEMFormer, Liao 2024, Placidi) requieren P_noisy como input, lo que exige haber ejecutado el circuito en hardware. El paradigma post-ejecución permite más información de entrada pero rompe la independencia del GEM respecto al QPU.

### TFM

**[SUPOSICIÓN TFM]** GEM predice Δ global (un único escalar), no Δ por observable. La fórmula de inferencia es: ⟨O⟩_mit = ⟨O⟩_noisy − Δ.

**Análisis de trade-off:**
- **Diferencia crítica con GTraQEM:** GTraQEM predice Δ *por observable* (por cada operador Pauli Z_i que se mide). El TFM predice un único Δ global para todos los observables. Esto es una simplificación que puede introducir error cuando distintos observables tienen sensibilidades de ruido distintas.
- El target escalar global simplifica la arquitectura pero reduce la expresividad. Para el alcance del TFM (máster), es aceptable, pero debe documentarse como limitación.
- La formulación residual (Δ en lugar de ⟨O⟩ directamente) está empíricamente justificada por GTraQEM. Mantener.

---

## 5. Función de Pérdida y Estrategia de Entrenamiento

### Lo que hacen los papers

**GTraQEM ✅:** MSE sobre Δ (error de expectation value). Entrenamiento end-to-end con Adam.

**QEMFormer ✅:** MSE + normalización del target. Entrenamiento conjunto de ambas ramas.

**Liao 2024 ✅:** MSE con regularización física (penaliza predicciones que violan restricciones de normalización de distribuciones cuánticas).

**Liao 2025 ✅:** Combina datos reales con datos aumentados (near-Clifford perturbados). La función de pérdida no requiere Statevector ground truth.

**[HECHO] Liao 2024:** La regularización física mejora la convergencia y reduce el error de extrapolación en un 15-20% respecto a MSE puro.

**Kim et al. ✅:** BCE (Binary Cross-Entropy) sobre la distribución de medición corregida.

**Lee & Park ✅:** MSE sobre la distribución corregida en subespacio.

### TFM

**[SUPOSICIÓN TFM]** MSE para el GEM (sobre Δ). MSE para el REM (sobre las matrices de confusión locales 2×2). GEM y REM se entrenan de forma desacoplada y paralela.

**Análisis de trade-off:**
- MSE para GEM está bien justificado (GTraQEM, QEMFormer).
- El desacoplamiento GEM/REM es una decisión arquitectónica sólida que evita contaminación cruzada de gradientes. No hay evidencia empírica contraria en los papers revisados.
- La normalización del target Δ no está especificada. GTraQEM y QEMFormer la incluyen — Δ puede variar varios órdenes de magnitud según el tipo de circuito. Sin normalización, el MSE puede estar dominado por circuitos de alta profundidad.

---

## 6. Generación de Datos de Entrenamiento y Ground Truth

Esta categoría es la más crítica para el TFM. Las diferencias respecto al SOTA son las más grandes.

### Lo que hacen los papers

**Método A — Statevector ideal en simulador (limitado a ~30 qubits):**
- GTraQEM y QEMFormer: usan simulación en circuitos pequeños para ground truth cuando no ejecutan en hardware real.
- Liu et al.: Statevector para circuitos de 3-16 qubits.
- [LIMITACIÓN CONOCIDA] Statevector escala exponencialmente en memoria: 15 qubits → 16 GB RAM, 20 qubits → 512 GB RAM.

**Método B — Near-Clifford Circuits (estado del arte para escalar):**
- Liao 2024 ✅: Training en near-Clifford circuits eficientemente simulables con el simulador estabilizador de Qiskit.
- Liao 2025 ✅: Near-Clifford circuits como base de data augmentation → permite escalar a más de 100 qubits sin RAM exponencial.
- Placidi ⚠️: Near-Clifford training en Quantinuum → generaliza a circuitos VQE/QAOA no-Clifford.
- Korolev ⚠️: Near-Clifford circuits como surrogate de training para VQE.
- [HECHO — Liao 2025] Un modelo entrenado en near-Clifford circuits generaliza a circuitos no-Clifford con degradación de rendimiento inferior al 5% en los benchmarks evaluados.

**Método C — Hardware real IBM (gold standard):**
- GTraQEM, QEMFormer, Q-Cluster, M3, Kim et al., Lee & Park: todos validan en hardware IBM real.
- [HECHO] GTraQEM ejecuta en ibm_kyiv y otros Eagle-family backends. QEM-Bench incluye datasets de ibm_kyiv (50q) e ibm_brisbane (63q).
- La brecha simulation-to-real (diferencia entre ruido simulado y ruido real) no está cuantificada en ninguno de los papers revisados, pero es reconocida como limitación en Liao 2024.

**Distribución de circuitos:**
- GTraQEM/QEMFormer (QEM-Bench): Trotterized TFIM, Random Clifford, QAOA — sin proporción explícita 80/20.
- Q-Cluster: QASMBench (22 low-entropy) + SupermarQ (7 high-entropy) = 29 benchmarks.
- Liu et al.: circuitos de algoritmos estándar + circuitos aleatorios.
- No hay un consenso en la literatura sobre proporciones específicas de circuitos random vs. estructurados.

### TFM

**[SUPOSICIÓN TFM]** Statevector ideal como ground truth. Limitado a máx 15 qubits. Ruido simulado con NoiseModel de Qiskit (no hardware real). Distribución: 80% random + 20% estructurados (QAOA/QFT/HEA).

**Análisis de trade-off — PROBLEMAS CRÍTICOS:**

1. **Barrera de 15 qubits:** El TFM no puede entrenar con circuitos >15 qubits porque el Statevector no cabe en RAM. El SOTA entrena rutinariamente con 50-127 qubits usando near-Clifford circuits o hardware real. Esto es la limitación más importante del TFM y debe documentarse explícitamente.

2. **Brecha simulation-to-real:** El NoiseModel de Qiskit es una aproximación de primer orden del ruido real. No captura coherent errors, crosstalk no-lineal, ni el drift step-like reportado por Hirasaki et al. Un modelo entrenado solo con datos simulados probablemente sobreestimará su rendimiento respecto al hardware real.

3. **Ausencia de near-Clifford circuits:** El TFM usa circuitos Clifford aleatorios (eficientemente simulables), pero no aprovecha la ventaja de near-Clifford circuits para escalar. Liao 2024/2025 y Placidi demuestran que near-Clifford es el método de elección del SOTA.

4. **Proporción 80/20 no justificada en la literatura:** El TFM debería ablacionar esta proporción o citarla como decisión de ingeniería sin precedente empírico.

---

## 7. Módulo REM (Readout Error Mitigation)

### Lo que hacen los papers

**M3 — Nation et al. (PRX Quantum 2021) ✅ — Fundamento matemático del REM:**
- Construye la matriz de asignación A (2^n × 2^n) de forma implícita mediante calibración: ejecuta n circuitos con todos los qubits en |0⟩ y n circuitos con todos los qubits en |1⟩ (total 2n circuitos).
- Asume que cada qubit es independiente de los demás (errores de lectura i.i.d. entre qubits) → la matriz A se factoriza como producto tensorial de matrices 2×2 locales.
- Resuelve A⁻¹ P_noisy = P_ideal usando GMRES solo en el subespacio de bitstrings observados → O(N·s) en lugar de O(2^{3n}).
- [HECHO] Escala a 65 qubits en hardware IBM real con tiempo de post-procesado <1 segundo. "Orders of magnitude less memory than direct factorization."

**Kim et al. (NJP 2022) ✅ — Primera DL para QREM:**
- Red neuronal entrenada con mediciones de circuitos de calibración (0 y 1 state) por qubit.
- La red aprende la función de corrección sin construir la matriz explícita.
- [HECHO] Supera a la inversión matricial estándar (incluyendo M3) cuando los errores de lectura son correlacionados entre qubits vecinos (no i.i.d.).
- [HECHO] Para ruido i.i.d., M3 y Kim et al. tienen rendimiento similar.

**Lee & Park (ML:ST 2023) ✅ — Escalabilidad via independencia condicional:**
- Explota la independencia condicional entre qubits no vecinos en la topología del chip.
- [HECHO] Reducción exponencial del tamaño de la red neuronal sin pérdida de precisión en hardware IBM de 7 y 13 qubits.
- Transfer learning entre dispositivos de distinta topología: añade factor constante de speedup.

**Q-Cluster (IEEE QCE 2025) ✅ — QREM sin ML como componente central:**
- La mitigación de readout se integra en el clustering (los errores de bitflip se modelan implícitamente como Hamming distance).
- No hay un módulo QREM separado explícito.

**GTraQEM / QEMFormer ✅:**
- No incluyen un módulo REM separado. Predicen directamente ⟨O⟩_mit sin separar ruido de puertas del ruido de lectura.

### TFM

**[SUPOSICIÓN TFM]** GNN predice matrices de confusión locales 2×2 por qubit → resuelve A⁻¹ P_noisy = P_ideal usando GMRES en el subespacio de bitstrings observados. Entrenado con circuitos triviales y solo ruido de lectura activado en el simulador.

**Análisis de trade-off:**
- La combinación GNN + GMRES es original: M3 usa matrices fijas de calibración; el TFM las predice dinámicamente vía GNN → más adaptativo a drift de hardware.
- La asunción de independencia por qubit (matrices 2×2 locales) es la misma que M3. Kim et al. demuestra que cuando hay correlaciones de lectura entre qubits vecinos, esta asunción falla. Para la topología Heavy-Hex, las correlaciones entre qubits adyacentes son no-negligibles.
- Lee & Park justifica empíricamente que la factorización local es correcta para qubits distantes (no vecinos directos). El TFM está alineado con esta evidencia.
- La GNN del REM opera sobre el grafo de topología del chip (no el DAG del circuito) → correcto: el readout error depende de la topología física, no del circuito ejecutado.

---

## 8. Concept Drift y Variabilidad Temporal

### Lo que hacen los papers

**Hirasaki et al. (APL 2023) ✅ — EVIDENCIA EMPÍRICA DIRECTA:**
> [HECHO] Las tasas de error en qubits superconductores IBM cambian de forma **escalonada** (step-like), no de forma gradual. Un cambio puede ocurrir en cualquier momento y persiste durante varios minutos o decenas de minutos. Múltiples eventos de salto pueden ocurrir en una misma sesión de experimentación. El drift no es predecible ni suave.
- Proponen post-selection de shots basado en correlaciones entre varianza del output y señal de error elevado.
- [HECHO] La recalibración diaria de IBM NO es suficiente para capturar estos saltos intra-sesión.

**Kim et al. (NJP 2022) ✅:**
> [HECHO] Los modelos de readout mitigation necesitan re-entrenamiento periódico conforme el hardware deriva. Sin re-entrenamiento, la precisión degrada visiblemente en 24-48 horas.

**Q-Cluster (IEEE QCE 2025) ✅:**
> [HECHO] El ExtraTrees regressor entrenado en IBM Eagle (ibm_kyiv, ibm_brisbane, ibm_brussels, ibm_strasbourg) generaliza a IBM Heron (ibm_torino, arquitectura diferente con CZ en lugar de ECR) con R²=0.886 sin reentrenamiento específico. Esto sugiere que los patrones de error tienen cierta robustez cross-arquitectura.

**Liao 2025 ✅:**
> [HECHO] El enfoque noise-agnostic (near-Clifford + data augmentation) reduce la sensibilidad al drift: el modelo no asume un noise model específico, por lo que cambios graduales en las tasas de error no lo degradan tan rápido como los enfoques calibración-dependientes.

**GTraQEM, QEMFormer, Liu et al., M3, Lee & Park:**
- No modelan explícitamente el Concept Drift temporal. Asumen que la calibración del momento de inferencia es representativa.

### TFM

**[SUPOSICIÓN TFM]** Drift simulado como decaimiento lineal de los error rates a lo largo de 10 días. El `day_index/10.0` como feature explícita señala el tiempo al modelo. Split OOD: Train días 1-5 | Val día 6 | Gap día 7 | Test OOD días 8-10.

**Análisis de trade-off — PROBLEMA IMPORTANTE:**
- **El drift lineal es factualmente incorrecto.** Hirasaki et al. demostró empíricamente que el drift es step-like. Entrenar con drift lineal simulado y evaluar en hardware real (con drift step-like) introduce una brecha de distribución que invalidaría la evaluación OOD.
- El `day_index` como feature explícita es una contribución original bien fundamentada teóricamente (con la evidencia de Hirasaki como motivación), pero no está validada empíricamente como feature efectivo en ningún paper.
- El split OOD temporal (5-1-1-3 días) es una decisión de ingeniería razonable y diferenciadora que ningún paper revisado replica exactamente — es una contribución metodológica del TFM.
- La brecha temporal forzada (día 7 sin datos) es un diseño conservador y correcto para simular el peor caso de drift.

---

## 9. Split Temporal y Metodología de Evaluación

### Lo que hacen los papers

**QEM-Bench (QEMFormer ✅):**
- 22 datasets estandarizados: 9 estándar + 9 zero-shot (circuitos no vistos en training) + 4 large-scale en hardware real (ibm_kyiv, ibm_brisbane).
- Métricas: MAE y RMSE sobre ⟨O⟩. Hellinger Fidelity para distribuciones completas.
- [HECHO] Zero-shot (generalización a circuitos no vistos) es la evaluación más exigente. QEMFormer supera a GTraQEM incluso en zero-shot.

**Q-Cluster ✅:**
- Split por tipo de distribución: low-entropy (22 benchmarks) vs. high-entropy (7 benchmarks).
- [HECHO] Q-Cluster solo es efectivo para low-entropy. Para high-entropy, M3 o ninguna mitigación son preferibles.
- 5 IBM QPUs distintas para robustez cross-device.

**GTraQEM ✅:**
- Evaluación en múltiples observables (Z, ZZ, etc.) en múltiples IBM backends.
- No hay split temporal — usa una sola sesión de calibración.

**Liao 2024 ✅:**
- Entrenamiento en días/sesiones pasadas → evaluación en nuevas sesiones.
- [HECHO] La degradación sin re-entrenamiento es moderada (5-10% en MSE) a lo largo de días, sugiriendo que los features de telemetría capturan parte del drift.

**M3, Kim et al., Lee & Park ✅:**
- No tienen split temporal — la calibración es puntual (ejecutada justo antes de la inferencia).

### TFM

**[SUPOSICIÓN TFM]** Split OOD temporal estricto: Train 1-5 | Val 6 | Gap 7 | Test 8-10. Evaluación en datos simulados (no en hardware real). Métrica implícita: MSE sobre ⟨O⟩ (no especificada explícitamente en los documentos del TFM).

**Análisis de trade-off:**
- El split OOD temporal es más conservador y más riguroso que lo que cualquier paper revisado propone. Es una fortaleza metodológica del TFM.
- **Problema:** La evaluación exclusiva en datos simulados no valida el rendimiento en hardware real. Ningún paper del SOTA se limita a simulación. Sin al menos una validación en hardware IBM real, el TFM no puede comparar directamente sus resultados con el SOTA.
- La métrica de evaluación (MAE/RMSE/Hellinger) no está especificada en los documentos actuales del TFM. QEM-Bench reporta MAE y RMSE sobre ⟨O⟩ — adoptar las mismas métricas facilitaría la comparación directa.

---

## 10. Composición del Dataset

### Lo que hacen los papers

| Paper | Circuitos usados | Proporción / Número |
|---|---|---|
| QEM-Bench (QEMFormer) ✅ | Trotterized TFIM, Random, QAOA | 22 datasets estandarizados |
| Q-Cluster ✅ | QASMBench + SupermarQ (22 low-entropy + 7 high-entropy) | 29 benchmarks |
| GTraQEM ✅ | Random + VQE + QAOA + QFT | Sin proporción explícita |
| Liu et al. ✅ | Algoritmos estándar (QFT, Grover, etc.) | 3-16 qubits |
| M3 ✅ | Circuitos de calibración (solo 0/1 states) | 2N circuitos |
| Kim et al. ✅ | Circuitos de calibración por qubit | 2N por qubit |
| Liao 2024/2025 ✅ | Near-Clifford circuits | No especificado |
| Placidi ⚠️ | Near-Clifford + VQE | ~1000 circuitos |
| Korolev ⚠️ | Near-Clifford para VQE/QAOA | No especificado |

**[HECHO] GTraQEM:** El modelo entrenado con circuitos random de IBM generaliza razonablemente a circuitos QAOA y VQE no vistos durante el training (zero-shot transfer).

**[HECHO] Q-Cluster:** Los métodos de QEM basados en distribuciones (M3, Q-Cluster, HAMMER) no mejoran o incluso empeoran la fidelidad para circuitos de alta entropía (Entropy >0.6). Solo las distribuciones de baja entropía se benefician de la redistribución.

**[HECHO] Placidi ⚠️:** El preentrenamiento en near-Clifford circuits + fine-tuning en hardware real mejora el rendimiento sobre el entrenamiento en hardware real solo, cuando la cantidad de datos de hardware real es limitada.

### TFM

**[SUPOSICIÓN TFM]** 80% random (Clifford/Unitary) + 20% estructurados (QAOA 33%, QFT 33%, HEA 33%). Profundidad: 5-50 layers. Qubits: 5-15.

**Análisis de trade-off:**
- La inclusión de QFT es inusual — QEM-Bench no la incluye como benchmark estándar. El TFM debería justificar su inclusión o considerar substituirla por Trotterized TFIM (que sí es benchmark estándar en QEM-Bench).
- La sustitución de Grover por HEA es correcta: Grover escala exponencialmente con qubits (mcx gate), mientras que HEA mantiene profundidad O(reps × n). La literatura (GTraQEM, Korolev) confirma HEA como benchmark estándar para QEM en circuitos variacionales.
- El rango 5-50 layers está bien justificado por el límite de señal útil (>50 layers → decoherencia casi total).
- La limitación a 5-15 qubits es consecuencia de la elección de Statevector ground truth, no una decisión de diseño favorable.

---

## 11. Recomendaciones para el TFM

Las recomendaciones se ordenan por impacto esperado (alto → bajo) y urgencia (antes vs. después de tener resultados).

### Cambios que recomiendo hacer ANTES de implementar los modelos

---

#### R1. Añadir nodo virtual QCR al Graph Transformer del GEM [ALTA PRIORIDAD]

**Qué cambiar:** En `gem_model.py` (aún no implementado), añadir un nodo virtual tipo QCR conectado a todos los nodos del DAG mediante aristas de atención separadas. El embedding del QCR se usa como readout global para la predicción de Δ.

**Por qué:** GTraQEM demuestra empíricamente que este nodo es el componente que permite contexto global sin message-passing. Es una diferencia pequeña de implementación con impacto grande de rendimiento.

**Coste:** Bajo — es una adición al modelo que no requiere cambiar el pipeline de datos.

---

#### R2. Corregir el modelo de Concept Drift de lineal a step-like [ALTA PRIORIDAD]

**Qué cambiar:** En `quantum_gen.py`, la función `build_noise_model_for_day()` (Bug 1 pendiente). En lugar de interpolar linealmente los error rates entre día 1 y día 10, modelar el drift como secuencia de saltos (escalones). Una implementación simple: asignar aleatoriamente 2-3 "eventos de salto" en el timeline de 10 días, donde el error rate cambia ±15-30% en un solo paso.

**Por qué:** Hirasaki et al. demuestra empíricamente que el drift en IBM es step-like, no gradual. Entrenar con drift lineal y evaluar en hardware con drift escalonado introduce una brecha de distribución que invalida la evaluación OOD.

**Coste:** Medio — requiere refactorizar `build_noise_model_for_day()` pero el cambio es localizado.

---

#### R3. Corregir el bug de zero-pad del qubit target [CRÍTICO — BUG ACTIVO]

**Qué cambiar:** En `quantum_gen.py`, la función que construye el vector de features. Para puertas de 1 qubit (rz, sx, x, id), el vector target qubit (features #6) debe ser `[0.0, 0.0, 0.0]`, no una copia del qubit control.

**Por qué:** El bug actual introduce información falsa (T1/T2/readout de un qubit que no existe en la puerta). Esto no solo es incorrecto semánticamente sino que puede confundir al modelo durante el entrenamiento.

**Coste:** Bajo — corrección de una línea en la implementación existente.

---

#### R4. Normalizar el target Δ por rango del observable [MEDIA PRIORIDAD]

**Qué cambiar:** En `train.py` (aún no implementado), normalizar Δ = (⟨O⟩_noisy − ⟨O⟩_ideal) / rango_observable antes de calcular el MSE. El rango del observable para un operador Pauli Z^n es [−n, +n].

**Por qué:** Sin normalización, el MSE estará dominado por circuitos de alta profundidad con Δ grande, y el modelo ignorará circuitos de poca profundidad. GTraQEM y QEMFormer incluyen esta normalización.

**Coste:** Bajo.

---

#### R5. Documentar explícitamente el paradigma pre-ejecución como diferenciador [MEDIA PRIORIDAD]

**Qué cambiar:** En la sección §4 del TFM (descripción arquitectónica), añadir una subsección explicando por qué el GEM NO incluye P_noisy como feature: el TFM opera en un paradigma pre-ejecución donde la predicción de Δ ocurre ANTES de ejecutar el circuito en el QPU. Esto es diferente de QEMFormer (post-ejecución). La ventaja operacional es que el GEM puede priorizar la ejecución de circuitos según su Δ esperado, sin necesitar una ejecución exploratoria previa.

**Por qué:** Sin esta documentación, un revisor puede interpretar la ausencia de P_noisy como omisión técnica en lugar de decisión de diseño consciente. Placidi demuestra que P_noisy es el feature dominante — el TFM necesita explicar por qué no lo usa.

**Coste:** Nulo en código, alto en claridad del TFM.

---

### Cambios que recomiendo considerar DESPUÉS de tener resultados experimentales

---

#### R6. Explorar near-Clifford circuits como alternativa al Statevector [BAJA PRIORIDAD a corto plazo]

**Qué cambiar:** Si el TFM tiene tiempo y los resultados con Statevector son insuficientes, reemplazar el 80% de circuitos random por near-Clifford circuits simulados con el simulador estabilizador de Qiskit (sin Statevector completo → sin límite de 15 qubits).

**Por qué:** Liao 2024/2025 y Placidi demuestran que near-Clifford training generaliza a circuitos no-Clifford. Esto eliminaría la barrera de 15 qubits y permitiría training más realista.

**Coste:** Alto — requiere cambiar el generador de circuitos, verificar que el simulador estabilizador de Qiskit-Aer funciona correctamente para near-Clifford, y repetir la generación del dataset completo.

---

#### R7. Incluir Hellinger Fidelity como métrica de evaluación del REM [MEDIA PRIORIDAD]

**Qué cambiar:** En `utils.py` y en los notebooks de evaluación, añadir la Hellinger Fidelity HF(P_mit, P_ideal) como métrica complementaria al MSE/RMSE. Q-Cluster usa HF como métrica primaria.

**Por qué:** MSE sobre ⟨O⟩ no captura si la distribución completa P_mit es fiel a P_ideal. HF mide la similitud distribucional directamente y es la métrica estándar para REM en la literatura reciente (Q-Cluster, QEM-Bench).

**Coste:** Bajo — una función en `utils.py`.

---

#### R8. Añadir Trotterized TFIM como tipo de circuito en el dataset [BAJA PRIORIDAD]

**Qué cambiar:** En `quantum_gen.py`, añadir generación de circuitos de simulación de Hamiltonianos tipo TFIM (Transverse-Field Ising Model) mediante Trotterización.

**Por qué:** Es el tipo de circuito más frecuente en QEM-Bench (QEMFormer) y en la literatura reciente. Su inclusión permitiría comparación directa con el SOTA en el mismo tipo de benchmark.

**Coste:** Medio — requiere implementar el generador TFIM, que es diferente en estructura al QAOA o HEA.

---

### Cosas que NO cambiaría

- **La separación GEM/REM:** Ningún paper muestra que la integración conjunta supere a la separación. GTraQEM y QEMFormer también distinguen implícitamente entre gate errors y readout errors.
- **El split OOD temporal (5-1-1-3):** Es más riguroso que lo que propone cualquier paper revisado. Mantener.
- **El vector de 19 dims del GEM:** Es competitivo con GTraQEM en cobertura de información. Solucionar los bugs pendientes (Bug 1) es suficiente.
- **El uso de GMRES en el REM:** Es el mismo solver que M3 (paper de IBM Research). La validación matemática es sólida.
- **HEA en lugar de Grover:** La sustitución ya fue validada en sesiones anteriores. HEA es el benchmark estándar para VQE en la literatura.

---

## 12. Lista de Lectura Prioritaria

Ordenada por urgencia antes de implementar los módulos correspondientes. ⭐ = leer esta semana.

| Prioridad | Paper | Razón | Antes de implementar |
|---|---|---|---|
| 1 ⭐ | **GTraQEM** (ICLR 2025) | Referencia directa del GEM. Arquitectura detallada del Graph Transformer + QCR | `gem_model.py` |
| 2 ⭐ | **M3 — Nation et al.** (PRX Quantum 2021) | Fundamento matemático del módulo REM. GMRES en subespacio | `rem_model.py` + `utils.py` |
| 3 ⭐ | **QEMFormer** (ICML 2025) | Nuevo SOTA a superar. Entender el paradigma post-ejecución para argumentar el diferenciador del TFM | §4 del TFM |
| 4 ⭐ | **Hirasaki et al.** (APL 2023) | Evidencia empírica del drift step-like. Crítico para corregir el modelo de drift lineal | `quantum_gen.py` Bug 1 |
| 5 | **Placidi et al.** ⚠️ (arXiv 2026) | Ablation study del vector de features. Justifica por qué P_noisy no es necesario en el paradigma pre-ejecución | §4 del TFM (argumentación) |
| 6 | **Kim et al.** (NJP 2022) | Primera DL para QREM. Baseline directo del REM | `rem_model.py` |
| 7 | **Lee & Park** (ML:ST 2023) | Independencia condicional en QREM. Justifica las matrices 2×2 locales | `rem_model.py` |
| 8 | **Liao et al. 2024** (Nature MI) | GNN en hardware IBM real hasta 100 qubits. Escalabilidad | Notebooks de evaluación |
| 9 | **Q-Cluster** (IEEE QCE 2025) | Paradigma alternativo de QREM. Baseline a comparar | Notebooks de evaluación |
| 10 | **Liu et al.** (Frontiers 2026) | GNN vs. CNN para predicción de circuitos. Feature engineering | §4 del TFM |
| 11 | **Das et al.** ⚠️ (arXiv 2025) | Contexto SOTA §3. Relevante para justificar DAG encoding | §3 del TFM |
| 12 | **Liao et al. 2025** (npj QI) | Noise-agnostic approach. Útil para la sección de limitaciones | §6 del TFM (limitaciones) |
| 13 | **Korolev et al.** ⚠️ (arXiv 2026) | Referencia para VQE/QAOA. Contexto de near-Clifford training | §3 del TFM |

---

*Documento generado el 2026-07-07. Actualizar cuando haya resultados experimentales.*
