# Guía de Notebooks de Análisis y Evaluación (notebooks/)

Este directorio contiene los Jupyter Notebooks organizados secuencialmente para el **Análisis Exploratorio de Datos (EDA)**, el **Prototipado del Pipeline** y la **Evaluación Crítica** de los resultados del TFM.

Siguiendo las mejores prácticas de MLOps de la industria (*Cookiecutter Data Science*), los notebooks están numerados cronológicamente según el flujo lógico de la investigación. **No contienen lógica de producción compleja**; en su lugar, importan los módulos modulares de `src/` y se enfocan puramente en el *storytelling* analítico, la visualización de métricas y la validación matemática.

---

## 📂 Índice Secuencial de Notebooks

### 📊 `01_eda_ibm_telemetry.ipynb` — Los datos del proyecto, explicados (EDA divulgativa)
* **Objetivo:** Auditar las fuentes de datos del pipeline Y servir de pieza de comunicación para audiencia no experta (tutor, tribunal). Cada gráfica lleva los ejes explicados en lenguaje llano y una conclusión en una frase.
* **Contenido:**
  * Qué aspecto tiene un circuito cuántico (dibujo real con `qc.draw('mpl')`) y su conversión a grafo — el input literal del GEM, con nodos coloreados por tipo de puerta.
  * El "parte médico" del chip **IBM Heron r2 (`ibm_kingston`)**: T1, T2, readout, gate_error por qubit — calibración sintética [SUPOSICIÓN] hasta disponer de credenciales IBM.
  * *Concept Drift* step-like: heatmap qubit×día (divergente, neutro = sin cambio) + series temporales con la zona de test OOD marcada como "examen final".
  * El dataset por dentro: los 5 splits y su rol, distribución del target Δ con la **franja de shot-noise** dibujada, tamaño/anchura de circuitos, y composición por tipo de puerta.
* **Valor para el Portfolio:** Demuestra la mentalidad de *Data Engineer* (auditar antes de entrenar) y la capacidad de comunicar resultados técnicos a audiencia general.

### 🧪 `02_quantum_circuit_prototyping.ipynb` — Tipos de Circuito y Estudio del Shot Noise
> Reenfocado (jul-2026) para no duplicar el 01, que ya dibuja circuitos y renderiza el grafo del extractor.
* **Objetivo:** Validar los 5 tipos de circuito lado a lado y justificar EMPÍRICAMENTE la elección de `LABEL_SHOTS`.
* **Contenido:**
  * Los 5 tipos (random, HEA, TFIM, QAOA, QFT) lado a lado: estructura, profundidad post-transpilación a base Heron, y distribución de salida ideal vs. ruidosa.
  * **Barrido de shots (1024 → 16384)** midiendo la estabilidad de la etiqueta Δ por tipo de circuito: cuantifica el suelo de shot-noise detectado en TAREA 8 y justifica el valor de `LABEL_SHOTS` del full run con evidencia propia.
  * Distinción explícita `LABEL_SHOTS` (etiquetas limpias) vs. `TRAIN_SHOTS` (inputs REM realistas) — ver `fallas_y_soluciones.md`.
* **Valor para el Portfolio:** Demuestra rigurosidad estadística — la decisión de shots deja de ser una [SUPOSICIÓN] y pasa a estar respaldada por un experimento reproducible.

### 🤖 `03_gem_transformer_evaluation.ipynb` — Evaluación del Módulo GEM
> ⚠️ Prerrequisito: etiquetas GEM regeneradas con readout DESACTIVADO (sub-ítem de TAREA 4). El dataset actual mezcla ambos ruidos en Δ.
* **Objetivo:** Analizar la convergencia y la precisión del Graph Transformer mitigando el ruido dinámico de las compuertas lógicas.
* **Contenido:**
  * Evaluación del rendimiento del regresor continuo sobre el conjunto de test *Out-Of-Distribution* (OOD) temporal (Días 8-10).
  * Curvas de aprendizaje, análisis de residuos matemáticos y distribución del error en la predicción del valor de desviación continuo ($\Delta$).
  * Pruebas de estrés variando la profundidad del circuito (de 5 a 50 capas) para identificar el punto de colapso del modelo.
* **Valor para el Portfolio:** Justifica la elección de la arquitectura basada en atención para capturar el paralelismo y la localidad cuántica sin violar los unitarios lógicos.

### 🕸️ `04_rem_gnn_matrix_free.ipynb` — Evaluación del Módulo REM y Subespacios
> ⚠️ Prerrequisito: dataset REM de circuitos triviales con SOLO readout (sub-ítem de TAREA 4). Aún no existe.
* **Objetivo:** Validar la escalabilidad matemática de la mitigación de errores de lectura sin colapsar la memoria RAM.
* **Contenido:**
  * Análisis del error de predicción de la GNN sobre las matrices locales marginales de confusión $2 	imes 2$.
  * Evaluación del solver iterativo de Krylov (**GMRES**) operando en la inversión libre de matrices (*Matrix-Free*).
  * Gráficas comparativas empíricas de complejidad espacial y consumo de memoria: Enfoque Clásico $\mathcal{O}(2^{3n})$ vs. Enfoque de Subespacio de Shots $\mathcal{O}(N)$.
* **Valor para el Portfolio:** Es la joya matemática del proyecto. Demuestra la viabilidad de escalar la mitigación a más de 100 qubits reales dentro de un entorno local estricto de 16GB de RAM.

### 🏁 `05_final_pipeline_tradeoffs.ipynb` — Acoplamiento de Inferencia y Trade-offs
* **Objetivo:** Evaluar el pipeline integrado final en condiciones de producción real y extraer los entregables académicos.
* **Contenido:**
  * Simulación del flujo completo de inferencia: Circuito del usuario $ightarrow$ Inferencia GEM ($\Delta$) $ightarrow$ Simulación con ruido total $ightarrow$ Inferencia REM (Limpieza de histograma) $ightarrow$ Ajuste algebraico final.
  * Análisis de compromiso (*Trade-off Analysis*) — **Ganancia de Fidelidad Cuántica vs. Latencia de Inferencia vs. Coste Computacional** — con barras agrupadas o *small multiples* (los radar comparan mal con pocas dimensiones).
  * Ablación de proporciones random/estructurados (70/30, 80/20, 90/10) — requiere proporciones configurables en `generate_dataset()` (sub-ítem TAREA 4).
  * Exportación automatizada de figuras y tablas en alta resolución para su inclusión directa en la memoria LaTeX del TFM.
* **Valor para el Portfolio:** Demuestra un enfoque de ingeniería de sistemas y MLOps maduro, orientado al rendimiento y la viabilidad del producto en entornos corporativos reales.

---

## 📐 Reglas de Oro para el Desarrollo en estos Notebooks

Para mantener la calidad de nivel de producción de este repositorio, el desarrollo dentro de este directorio debe regirse por los siguientes principios estrictos:

1. **Cero Funciones Kilométricas:** Si una celda requiere más de 20 líneas de manipulación de datos o tensores, esa lógica debe ser refactorizada, trasladada a `src/utils.py` o `src/dataset.py`, e importada en el notebook.
2. **Reproducibilidad Absoluta:** La primera celda de código de cada notebook debe inicializar las semillas de aleatoriedad global importadas desde `src/config.py`. Cualquier ejecución completa (*Run All*) debe replicar exactamente las mismas curvas y figuras.
3. **Documentación Narrativa Extensiva:** Cada sección de código debe estar precedida por bloques de Markdown detallados que expliquen el fundamento matemático (álgebra lineal, operadores de Kraus, matrices de densidad o subespacios lineales) de la operación. Las gráficas deben incluir etiquetas claras en sus ejes, leyendas y conclusiones analíticas escritas.
