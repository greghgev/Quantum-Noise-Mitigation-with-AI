# Ideas Futuras — TFM-Quantum

> Extensiones y mejoras que **no se implementarán en el TFM actual** pero que:
> 1. Pueden ser trabajo futuro si hay tiempo.
> 2. Deben citarse en la sección §7 "Trabajo Futuro" del TFM como líneas de investigación abiertas.
>
> Cada idea incluye por qué es valiosa, qué coste tendría implementarla, y en qué parte del SOTA está respaldada.

---

## IDEA-0 — El módulo REM completo (⭐ la línea futura principal) — ANEXO CONSOLIDADO

> **Este es el único lugar del repositorio donde vive la información del módulo REM** (Readout
> Error Mitigation), retirado del alcance del TFM en jul-2026 por acuerdo con el tutor. Aquí se
> consolida TODO: diseño, matemática, estrategia de entrenamiento, métricas, referencias y
> análisis SOTA — movidos desde CLAUDE.md, fallas_y_soluciones.md y comparative_analysis.md.
> El código esqueleto (rem_model.py, train_rem.py, rem_config.yaml) fue eliminado del árbol;
> recuperable del historial git si se retoma.

### Qué es

La segunda mitad del diseño original del TFM: un corrector del error del sensor de lectura que
actúa DESPUÉS de ejecutar en el QPU. Una GNN predice matrices de confusión locales **2×2 por
qubit** (dinámicas: recalculadas con la calibración del día) y la corrección se aplica con un
solver iterativo **GMRES matrix-free** que opera solo en el subespacio de bitstrings observados.
El pipeline completo original era: `Circuito → GEM predice Δ_puertas → QPU → REM limpia
histograma → ⟨O⟩_mit = ⟨O⟩_limpio − Δ`.

### La matemática (el argumento de escalabilidad)

- **El problema:** la mitigación de lectura clásica invierte una matriz de confusión global de
  2ⁿ×2ⁿ → O(2^{3n}). Inviable a partir de ~15-20 qubits.
- **Paso 1 — de global a local:** el error del sensor es mayoritariamente local, así que la
  matriz global es (aprox.) el producto tensorial de matrices locales: A = A₁ ⊗ A₂ ⊗ … ⊗ Aₙ.
  La GNN predice solo las Aᵢ (2×2 cada una; 4×4 para pares con crosstalk). El producto tensorial
  JAMÁS se construye.
- **Paso 2 — el subespacio de los shots:** con S shots, la máquina devuelve como máximo S
  bitstrings distintos; los otros 2ⁿ−S estados tienen cero cuentas y son irrelevantes. El sistema
  Ax = b se proyecta solo sobre los estados observados.
- **Paso 3 — matrix-free:** GMRES (o BiCGSTAB, `scipy.sparse.linalg`) no necesita la matriz A,
  solo una función v → A·v que aplica las matrices locales secuencialmente sobre los bitstrings
  observados. Complejidad O(N), converge en pocas iteraciones.

### Estrategia de entrenamiento (desacoplada — requiere simulador)

El REM se entrenaría con un dataset PROPIO, distinto del GEM: circuitos triviales (preparar
estados base y medir) con SOLO ruido de lectura activado en el simulador (`TRAIN_SHOTS=1024`,
bajo a propósito: entrenar con histogramas casi perfectos sobreajustaría a distribuciones que
el hardware real nunca produce). Nota clave: **aislar el ruido de lectura solo es posible en
simulador** — el hardware real siempre devuelve ambos ruidos mezclados. En inferencia, el
acoplamiento GEM+REM ocurriría solo en producción (secuencial), manteniendo el entrenamiento
paralelo e independiente.

### Métricas de evaluación

| Métrica | Qué mide |
|---|---|
| Hellinger Fidelity (HF) | Parecido entre P_mit y P_ideal (0–1; 1 = idénticos) |
| Total Variation Distance (TVD) | Diferencia absoluta total entre distribuciones (0 = perfecto) |
| Ratio de mejora | HF_mit / HF_noisy (>1 = el REM ayuda) |

### Qué queda preparado en el proyecto

- Cada muestra `.pt` del dataset ya guarda `ideal_probs` y `noisy_probs` (distribuciones
  completas) — los inputs/targets que el REM necesitaría.
- La feature de readout_error por qubit ya está en la calibración descargada a diario.
- Extensión natural: readout asimétrico (P(1|0) ≠ P(0|1), campos `prob_meas1_prep0`/
  `prob_meas0_prep1` confirmados en la API de IBM) — la matriz 2×2 dejaría de ser simétrica.

### Análisis SOTA (condensado de comparative_analysis §7)

- **M3 es el fundamento directo**: mismo esquema subespacio+matrix-free, pero con matrices de
  calibración FIJAS. El REM las haría DINÁMICAS (predichas por GNN según el estado del chip) —
  ese es el hueco de contribución.
- **La factorización local está justificada teóricamente**: Lee & Park demuestran que los errores
  de lectura entre qubits no vecinos son condicionalmente independientes (reducción exponencial
  del tamaño de red sin pérdida relevante).
- **La no-linealidad justifica la GNN**: Kim et al. demuestran que el readout tiene componentes
  no lineales que la inversión lineal clásica no captura y una red sí.
- **Limitación conocida del diseño 2×2**: Guo & Yang (tensor networks, arXiv:2606.25974)
  cuantifican que los modelos no correlacionados pierden errores de lectura correlacionados
  entre qubits; un MPO los captura con coste ~lineal. Sería la evolución natural del REM.

### Referencias completas (movidas de CLAUDE.md §8)

- **M3 ⭐ — Nation, Kang, Sundaresan, Gambetta (IBM Research), PRX Quantum 2, 040326, 2021.**
  "Scalable mitigation of measurement errors on quantum computers." DOI: 10.1103/PRXQuantum.2.040326.
  Solver iterativo precondicionado en el subespacio de bitstrings observados, sin construir la
  matriz 2ⁿ×2ⁿ; converge en O(1) pasos con órdenes de magnitud menos memoria. PDF en `doc/SOTA/papers/`.
- **Kim, Oh, Chong, Hwang, Park — New Journal of Physics 24, 073009, 2022.** "Quantum readout
  error mitigation via deep learning." DOI: 10.1088/1367-2630/ac7b3d. Primera aplicación de DL a
  la corrección de lectura; supera a la inversión lineal en MSE/KL/infidelidad (IBM 5q). PDF en `papers/`.
- **Lee & Park — Machine Learning: Science and Technology 4, 045051, 2023.** "Scalable quantum
  measurement error mitigation via conditional independence and transfer learning."
  DOI: 10.1088/2632-2153/ad1007. Independencia condicional → reducción exponencial; transfer
  learning entre dispositivos (IBM 7q/13q). PDF en `papers/`.
- **Guo & Yang — arXiv:2606.25974 (2026, preprint).** "Tensor network characterization and
  mitigation of readout errors." MPO para errores de lectura correlacionados. PDF en `papers/`.

**Coste de retomarlo:** alto — módulo completo (dataset propio + modelo + entrenamiento +
evaluación). El diseño de arriba está cerrado; la implementación partiría de cero (el esqueleto
borrado era solo docstrings).

---

## IDEA-1 — Evaluación cross-device (múltiples QPUs de IBM)

**Qué es:**
El TFM se entrena y evalúa únicamente en `ibm_kingston` (Heron r2, 156 qubits). Los papers del SOTA (Q-Cluster, GTraQEM, QEMFormer) testean en 3-5 QPUs distintas para demostrar que el modelo no está sobreajustado a una sola máquina.

Una extensión natural sería evaluar el modelo entrenado en `ibm_kingston` directamente en `ibm_pittsburgh` (Heron r3, revisión más nueva) o en un backend Nighthawk (topología cuadrada, arquitectura diferente) sin reentrenamiento.

**Por qué es valiosa:**
- Q-Cluster demostró empíricamente que un ExtraTrees entrenado en Eagle generaliza a Heron con R²=0.886 sin reentrenamiento. Si el GEM del TFM tiene propiedades similares, sería un resultado fuerte.
- Demuestra robustez real del sistema frente a cambios de hardware, que es el escenario de producción real.

**Coste:**
Bajo en código (el pipeline de inferencia ya existe), pero requiere acceso a créditos adicionales de IBM Quantum para ejecutar circuitos en dos máquinas. El principal obstáculo es presupuestario, no técnico.

**Respaldo en la literatura:**
Q-Cluster [Patil et al., IEEE QCE 2025], GTraQEM [Bao et al., ICLR 2025].

---

## IDEA-2 — Near-Clifford circuits para eliminar la barrera de 15 qubits

**Qué es:**
Actualmente el TFM está limitado a circuitos de máximo 15 qubits porque el ground truth se calcula con `Statevector` (RAM exponencial: 15 qubits = 16 GB). Near-Clifford circuits son circuitos que están "cerca" de ser puramente Clifford — pueden simularse eficientemente con el simulador estabilizador de Qiskit sin RAM exponencial, incluso para 50-100 qubits.

Si se entrena el GEM con near-Clifford circuits en lugar de Statevector puro, se podría escalar a 50+ qubits sin cambiar el hardware de desarrollo.

**Por qué es valiosa:**
- Liao et al. 2024 (Nature MI) y Liao et al. 2025 (npj QI) y Placidi et al. (Quantinuum) demuestran que modelos entrenados en near-Clifford generalizan bien a circuitos no-Clifford (degradación <5% en los benchmarks evaluados).
- Eliminar la barrera de 15 qubits transformaría el TFM de un sistema de prueba de concepto a un sistema competitivo en escala con el SOTA.

**Coste:**
Alto. Requiere:
1. Modificar `quantum_gen.py` para generar near-Clifford circuits (añadir perturbaciones pequeñas a circuitos Clifford puros).
2. Verificar que el simulador estabilizador de Qiskit-AER funciona correctamente para estos circuitos.
3. Repetir la generación completa del dataset.
4. Revisitar la validación del ground truth (ya no es Statevector exacto — es una aproximación).

**Respaldo en la literatura:**
Liao et al. 2024 [Nature MI], Liao et al. 2025 [npj QI], Placidi et al. ⚠️ [arXiv:2601.14226].

---

## IDEA-3 — Δ por observable en lugar de Δ global

**Qué es:**
El GEM del TFM predice un único Δ escalar para todo el circuito, que se resta del valor esperado ruidoso. GTraQEM predice un Δ distinto para cada observable que el usuario quiere medir (⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₁Z₂⟩, etc.).

Una extensión sería modificar el GEM para que prediga un vector de Δ con un componente por observable. El nodo virtual QCR generaría un embedding multidimensional que el MLP final proyectaría en tantos escalares como observables tenga el circuito.

**Por qué es valiosa:**
Distintos observables tienen sensibilidades de ruido distintas. Una puerta que afecta al qubit 1 introduce más error en ⟨Z₁⟩ que en ⟨Z₂⟩. Un Δ por observable permite correcciones más precisas, especialmente para circuitos con muchos qubits y múltiples observables simultáneos.

**Coste:**
Medio. Requiere:
1. Modificar la capa de salida del GEM (1 escalar → N escalares).
2. Modificar el ground truth del dataset para almacenar Δ por observable, no solo Δ global.
3. Posiblemente modificar la función de pérdida (MSE por observable vs. MSE global).

**Respaldo en la literatura:**
GTraQEM [Bao et al., ICLR 2025].

---

## IDEA-4 — Fine-tuning con datos reales de IBM

**Qué es:**
El modelo se pre-entrena con datos simulados (NoiseModel de Qiskit-AER) y luego se hace un fine-tuning con un conjunto pequeño de datos reales ejecutados en `ibm_kyiv`. Este enfoque (pre-train en simulación + fine-tune en real) es el estándar en transfer learning de ML-QEM.

**Por qué es valiosa:**
Placidi et al. (Quantinuum) demuestran que preentrenamiento en simulador + fine-tuning en hardware real mejora el rendimiento sobre entrenamiento en hardware real solo, cuando los datos reales son escasos (que es siempre el caso por coste de QPU). Mitigaría la brecha simulation-to-real.

**Coste:**
Medio. La arquitectura no cambia. Solo requiere:
1. Generar un dataset pequeño real (50-100 circuitos en `ibm_kingston`).
2. Un bucle de fine-tuning con learning rate bajo.
3. Verificar que no hay overfitting a los pocos datos reales (early stopping).

**Respaldo en la literatura:**
Placidi et al. ⚠️ [arXiv:2601.14226], Liao et al. 2024 [Nature MI].

---

## IDEA-5 — Añadir P_noisy como feature en un modo post-ejecución opcional

**Qué es:**
El GEM actual opera en modo pre-ejecución: predice Δ sin ver el resultado del QPU. Una extensión sería añadir un "modo post-ejecución" donde, si el usuario ya tiene P_noisy disponible, se lo pase al modelo como feature adicional.

Esto convertiría al GEM en un modelo híbrido: puede operar pre-ejecución (menos información, más rápido) o post-ejecución (más información, más preciso), según el contexto del usuario.

**Por qué es valiosa:**
Placidi et al. demuestra que P_noisy es el feature dominante para la predicción de error. QEMFormer la usa como input principal. Añadir esta capacidad opcional al TFM lo haría competitivo en precisión con QEMFormer sin perder la capacidad de operar pre-ejecución.

**Coste:**
Medio-alto. Requiere:
1. Dos modos de forward pass en el GEM (con y sin P_noisy).
2. Modificar el dataset para almacenar P_noisy como feature adicional.
3. Entrenamiento separado (o con dropout en el feature P_noisy para que el modelo aprenda a funcionar sin él).

**Respaldo en la literatura:**
QEMFormer [Bao et al., ICML 2025], Placidi et al. ⚠️ [arXiv:2601.14226].

---

## IDEA-6 — Regularización física en la función de pérdida

**Qué es:**
Añadir a la función de pérdida del GEM términos de regularización que penalicen predicciones físicamente imposibles. Por ejemplo: la distribución de probabilidad corregida debe sumar 1; los valores esperados de observables de Pauli deben estar en [-1, +1]; la fidelidad entre P_mit y P_ideal nunca puede ser negativa.

**Por qué es valiosa:**
Liao et al. 2024 demuestra que la regularización física mejora la convergencia y reduce el error de extrapolación en un 15-20% respecto a MSE puro.

**Coste:**
Bajo. Son términos adicionales en la función de pérdida del `train.py` ya existente.

**Respaldo en la literatura:**
Liao et al. 2024 [Nature MI, DOI: 10.1038/s42256-024-00927-2].

---

## IDEA-7 — Ablación del nodo QCR vs. average pooling vs. max pooling

**Qué es:**
El TFM usa el nodo virtual QCR para obtener el embedding global del circuito. Una ablación sistemática compararía QCR vs. average pooling sobre todos los nodos vs. max pooling, para confirmar que QCR es la mejor opción en este dominio.

**Por qué es valiosa:**
GTraQEM afirma que QCR supera al pooling pero no publica una ablación detallada en el paper. Si el TFM la hace, es una contribución metodológica pequeña pero honesta.

**Coste:**
Bajo. Solo requiere entrenar el mismo modelo tres veces con distintos readout heads y comparar métricas.

**Respaldo en la literatura:**
GTraQEM [Bao et al., ICLR 2025].
