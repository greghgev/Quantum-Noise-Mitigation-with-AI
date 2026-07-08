# Idea General del TFM — Mitigación de Ruido Cuántico con IA

> Este documento explica en lenguaje claro qué hace el TFM, cómo lo hace, por qué es diferente al estado del arte, y qué decisiones de diseño se han tomado y por qué.
> Es el documento de referencia conceptual. Los detalles técnicos están en `CLAUDE.md` y `dataset_info.md`.

---

## 1. El problema que resuelve el TFM

Los ordenadores cuánticos de IBM cometen errores constantemente. Hay dos tipos de errores distintos:

**Error de puertas:** durante el circuito, las operaciones cuánticas (puertas lógicas) no son perfectas. El chip tiene temperatura, vibraciones, imperfecciones físicas. Cada puerta introduce una pequeña desviación acumulada.

**Error de lectura:** al final del circuito, cuando el chip mide el resultado, el sensor también se equivoca. A veces lee un 0 cuando debería leer un 1, o al revés.

Estos dos errores son independientes entre sí, tienen orígenes físicos distintos, y requieren correcciones distintas.

La solución clásica era ejecutar el mismo circuito 50-100 veces y hacer la media (técnicas ZNE, PEC). Es precisa pero carísima: cada ejecución en IBM cuesta tiempo de máquina real, que empresas y universidades pagan.

**El TFM consigue ejecutar el circuito una sola vez y obtener un resultado corregido.**

---

## 2. El pipeline de mitigación — paso a paso

```
[CIRCUITO EN PAPEL]
        │
        ▼
┌───────────────────────────────┐
│  MÓDULO GEM                   │
│  (Gate Error Mitigation)      │
│                               │
│  Entrada: el circuito + la    │
│  calibración del chip ese día │
│                               │
│  Proceso: Graph Transformer   │
│  analiza la estructura del    │
│  circuito como un grafo       │
│                               │
│  Salida: predice Δ            │
│  (el error esperado de las    │
│  puertas, un número escalar)  │
└───────────────────────────────┘
        │
        │ Δ predicho
        ▼
┌───────────────────────────────┐
│  EJECUCIÓN EN IBM QPU         │
│                               │
│  El circuito se ejecuta       │
│  INTACTO, sin modificaciones  │
│  → resultado ruidoso ⟨O⟩noisy │
└───────────────────────────────┘
        │
        │ histograma de mediciones ruidosas
        ▼
┌───────────────────────────────┐
│  MÓDULO REM                   │
│  (Readout Error Mitigation)   │
│                               │
│  Entrada: histograma ruidoso  │
│  + topología física del chip  │
│                               │
│  Proceso: GNN predice el      │
│  error de lectura por qubit   │
│  → corrección algebraica      │
│  (GMRES en subespacio)        │
│                               │
│  Salida: ⟨O⟩noisy corregido   │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  RESULTADO FINAL              │
│                               │
│  ⟨O⟩_mit = ⟨O⟩_noisy − Δ    │
│                               │
│  Resultado corregido con una  │
│  sola ejecución en el QPU     │
└───────────────────────────────┘
```

---

## 3. Por qué es diferente al estado del arte

### El paradigma post-ejecución (QEMFormer, el nuevo SOTA)

El modelo más fuerte actualmente (QEMFormer, ICML 2025) funciona así:
1. Ejecutas el circuito en el QPU.
2. Obtienes el histograma ruidoso (lo que IBM llama P_noisy).
3. Se lo das al modelo como input.
4. El modelo lo corrige todo de golpe — errores de puertas y de lectura mezclados.

Es muy preciso porque tiene toda la información disponible. El problema es que solo puede corregir después de ejecutar, y no separa los dos tipos de error.

### El paradigma del TFM (pre-ejecución del GEM + post-ejecución del REM)

El GEM predice el error de puertas **antes** de ejecutar el circuito, mirando únicamente el diseño del circuito y la calibración del chip ese día. Cuando ejecutas y obtienes el resultado ruidoso, ya tienes Δ calculado y solo tienes que restárselo.

**Esto abre dos capacidades que QEMFormer no tiene:**

**Capacidad 1 — Filtro de calidad previo a la ejecución:**
Si tienes 100 circuitos candidatos y solo presupuesto para ejecutar 20 en IBM, el GEM puede estimar el error esperado de cada uno sin gastar ni un segundo de QPU. Seleccionas los 20 con menor error predicho. Esto es imposible con enfoques post-ejecución: tendrías que ejecutarlos todos antes de saber cuáles valen la pena.

**Capacidad 2 — Separación limpia de los dos tipos de error:**
El GEM solo aprende el error de puertas (REM desactivado durante su entrenamiento). El REM solo aprende el error de lectura (GEM separado). Esta separación tiene valor científico: permite estudiar cada fuente de ruido de forma independiente, algo que QEMFormer no permite porque los mezcla en un solo predictor.

> **El TFM no sacrifica la corrección post-ejecución**: el REM opera después de la ejecución con toda la información del histograma. La diferencia es que la corrección de errores de puertas se calcula antes, de forma independiente y más eficiente.

---

## 4. Dataset — decisiones de diseño

### Hardware objetivo
IBM Heron r2 (`ibm_kingston`), 156 qubits, puerta 2-qubit nativa CZ. Subgrafos de 5 a 15 qubits (límite impuesto por la memoria RAM necesaria para calcular el ground truth con Statevector).

> Nota: el proyecto arrancó sobre IBM Eagle (`ibm_kyiv`, 127q, Heavy-Hex), pero IBM retiró toda la familia Eagle entre 2023 y abril de 2026. La migración a Heron se documenta en ROADMAP.md (TAREA 3b).

### Tipos de circuitos y proporciones

| Tipo | Proporción | Rol |
|---|---|---|
| Random (Clifford/Unitary) | 80% | Base de entrenamiento — obliga al modelo a aprender la física pura del ruido sin memorizar patrones de algoritmos |
| HEA (Hardware-Efficient Ansatz / VQE) | ~7% | Circuitos variacionales parametrizados — benchmark estándar en QEM |
| Trotterized TFIM | ~7% | Simulación de Hamiltoniano de Ising — benchmark estándar de QEM-Bench (QEMFormer) |
| QAOA | ~6% → **reservado zero-shot** | No aparece en training; se usa solo en test zero-shot |
| QFT | variable → **reservado zero-shot** | No aparece en training; contribución original del TFM |

**Nota sobre la proporción 80/20:** La división entre circuitos random y estructurados no tiene precedente empírico en la literatura. Se ablacionará probando 70/30 y 90/10 y se reportarán los resultados. La proporción final se elegirá según el rendimiento en validación.

### Profundidad de los circuitos
Entre 5 y 50 capas. Menos de 5: apenas hay ruido medible. Más de 50: la señal cuántica queda destruida por la decoherencia y el modelo aprende ruido puro estocástico, sin información útil que recuperar.

### Por qué QFT es una contribución original a la SOTA

El SOTA de ML-QEM (QEM-Bench, Q-Cluster, GTraQEM) evalúa sobre: Random, QAOA, VQE/HEA, Trotterized TFIM, GHZ, Bernstein-Vazirani. **QFT no aparece en ningún benchmark estándar de ML-QEM.**

Los circuitos QFT tienen características únicas: alta profundidad, entrelazamiento denso y determinista, y alta entropía en la distribución de salida. Q-Cluster demostró empíricamente que los métodos de redistribución de distribuciones no funcionan bien para circuitos de alta entropía. Sin embargo, Q-Cluster opera sobre distribuciones (P_noisy), no sobre grafos de circuitos como el GEM. Nadie ha probado si la predicción de Δ mediante Graph Transformer funciona o no para QFT. El TFM llena ese hueco.

### El split temporal (Concept Drift)

El hardware de IBM no se recalibra de forma gradual — los error rates cambian de forma escalonada (saltos bruscos que persisten minutos u horas), según evidencia empírica de Hirasaki et al. (Applied Physics Letters, 2023). El TFM simula este comportamiento con saltos aleatorios en la línea temporal de 10 días.

| Días | Rol |
|---|---|
| 1 – 5 | Training |
| 6 | Validación |
| 7 | Brecha temporal forzada (sin datos) |
| 8 – 10 | Test OOD estricto |

El modelo se considera apto para producción si mitiga correctamente la calibración del día 10 habiendo visto solo hasta el día 5.

---

## 5. Evaluación zero-shot

Zero-shot significa: el modelo no ve estos circuitos durante el entrenamiento. Si funciona bien en ellos, demuestra que aprendió la física del ruido, no los patrones de circuitos específicos.

| Set de evaluación | Circuitos | Por qué |
|---|---|---|
| In-distribution | Random + HEA + TFIM (no vistos en train) | Evaluación estándar |
| Zero-shot A | **QAOA completo** | Es el benchmark más citado en el SOTA. Generalizar a QAOA sin entrenarlo es el resultado más valorado por revisores |
| Zero-shot B | **QFT completo** | Es la contribución original del TFM. Alta entropía, nunca evaluado en ML-QEM con Graph Transformer |

**Por qué QAOA para zero-shot y no HEA:** QAOA y HEA son familias distintas de circuitos variacionales. Entrenar con HEA y generalizar a QAOA demuestra que el modelo captura la física general de los circuitos variacionales, no solo los patrones de HEA.

---

## 6. Métricas de evaluación

Se reportarán todas las métricas relevantes para permitir comparación directa con el SOTA.

### Para el GEM (predicción de errores en valores esperados)

| Métrica | Qué mide |
|---|---|
| MAE | Error medio absoluto en ⟨O⟩ |
| RMSE | Error cuadrático medio (penaliza errores grandes) |
| R² | Correlación entre Δ predicho y Δ real (1.0 = perfecto) |
| Mejora relativa | \|⟨O⟩noisy − ⟨O⟩ideal\| vs \|⟨O⟩mit − ⟨O⟩ideal\| |

### Para el REM (corrección de distribuciones de probabilidad)

| Métrica | Qué mide |
|---|---|
| Hellinger Fidelity (HF) | Parecido entre P_mit y P_ideal (0 a 1; 1 = idénticos) |
| Total Variation Distance (TVD) | Diferencia total entre distribuciones (0 = perfecto) |
| Ratio de mejora | HF_mit / HF_noisy (>1 significa que el REM ayuda) |

Estas métricas se calculan separadamente para:
- Circuitos in-distribution (random + HEA + TFIM)
- Zero-shot A (QAOA)
- Zero-shot B (QFT)
- Cada día del split temporal (para visualizar la degradación por Concept Drift)

---

## 7. Validación en hardware real IBM

Todo el entrenamiento del TFM se realiza con datos de **ruido simulado** (Qiskit AER NoiseModel). El simulador es una aproximación de primer orden: no captura todos los tipos de ruido real (crosstalk no lineal, pulsos imperfectos, fluctuaciones de frecuencia).

Para que los resultados del TFM sean creíbles, se planifica una validación mínima en hardware IBM real:

- **Número de circuitos:** 20-30 circuitos de distintos tipos y profundidades.
- **Objetivo:** comprobar si el modelo entrenado con datos simulados generaliza al ruido real de `ibm_kingston`.
- **Resultado esperado si generaliza:** el MAE en hardware real es comparable al MAE en test simulado → el pipeline de simulación es suficientemente fiel.
- **Resultado si no generaliza:** se documenta como limitación en §6 del TFM y se propone fine-tuning con datos reales como trabajo futuro.

Sin esta validación, cualquier revisor puede preguntar "¿y en hardware real?" sin que haya respuesta. Con ella, el TFM tiene una respuesta empírica honesta en cualquiera de los dos casos.

---

## 7. Limitación conocida — Δ global vs. Δ por observable

El GEM predice **un único Δ escalar** para todo el circuito. Ese Δ se resta del valor esperado ruidoso ⟨O⟩_noisy para obtener el resultado mitigado.

El problema: en un circuito real, si mides varios observables distintos (por ejemplo ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₁Z₂⟩), cada uno tiene una sensibilidad al ruido diferente. La puerta que afecta al qubit 1 introduce más error en ⟨Z₁⟩ que en ⟨Z₂⟩. Un Δ global único no puede capturar esas diferencias.

**Lo que hace GTraQEM (el referente):** predice un Δ distinto por cada observable que el usuario quiere medir. Más preciso, más complejo de implementar.

**Lo que hace el TFM:** predice un Δ promedio para todos los observables del circuito. Es una simplificación que reduce la complejidad de la arquitectura, especialmente útil cuando el TFM se usa como filtro previo (Capacidad 1 del §3) más que como corrector de precisión máxima.

Esta es una limitación real que se documenta honestamente en el TFM. Ver `IDEAS_FUTURAS.md` para la extensión posible.

---

## 8. Entrenamiento — principios clave

- **GEM y REM se entrenan de forma desacoplada y paralela.** El GEM se entrena con ruido solo de puertas (readout desactivado en el simulador). El REM se entrena con circuitos triviales y solo ruido de lectura activado. Esto evita que los gradientes de un módulo contaminen al otro.

- **El target del GEM es el residuo Δ, no el valor absoluto.** Predecir la diferencia (error) en lugar del valor absoluto estabiliza el entrenamiento. Δ es numéricamente pequeño y varía menos entre circuitos que ⟨O⟩_noisy.

- **Δ se normaliza antes de calcular el loss.** Se divide por el rango del observable medido para que circuitos de distinta escala contribuyan por igual al entrenamiento.

- **El Graph Transformer del GEM incluye un nodo virtual QCR** (Quantum Circuit Representative). Este nodo ficticio tiene conexión directa con todos los nodos del circuito y actúa como resumen global del circuito completo, permitiendo que cualquier puerta "vea" el contexto global sin necesidad de message-passing entre nodos.
