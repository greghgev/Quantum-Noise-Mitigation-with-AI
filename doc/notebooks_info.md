# Guía de Notebooks de Análisis y Evaluación (notebooks/)

Este directorio contiene los Jupyter Notebooks organizados secuencialmente para el **Análisis Exploratorio de Datos (EDA)**, el **Prototipado del Pipeline** y la **Evaluación Crítica** de los resultados del TFM. 

Siguiendo las mejores prácticas de MLOps de la industria (*Cookiecutter Data Science*), los notebooks están numerados cronológicamente según el flujo lógico de la investigación. **No contienen lógica de producción compleja**; en su lugar, importan los módulos modulares de `src/` y se enfocan puramente en el *storytelling* analítico, la visualización de métricas y la validación matemática.

---

## 📂 Índice Secuencial de Notebooks

### 📊 `01_eda_ibm_telemetry.ipynb` — Análisis Exploratorio y Drift Temporal
* **Objetivo:** Analizar el comportamiento físico y la degradación diaria del hardware cuántico NISQ.
* **Contenido:**
  * Ingesta e inspección del historial de calibración de 10 días de la arquitectura *Heavy-Hex* del chip **IBM Eagle (127 qubits)**.
  * Análisis de series temporales del decaimiento de los tiempos de relajación térmica ($T_1$) y decoherencia de fase ($T_2$).
  * Visualización del *Concept Drift* (cambio de distribución del ruido) en los errores empíricos de las compuertas y tasas de fallo de los sensores de lectura.
* **Valor para el Portfolio:** Demuestra la mentalidad de *Data Engineer* al auditar las fuentes de datos y caracterizar la corrupción de la señal física antes de alimentar las redes neuronales.

### 🧪 `02_quantum_circuit_prototyping.ipynb` — Prototipado del Dataset y Simulación
* **Objetivo:** Validar la lógica del generador sintético y la inyección controlada de ruido cuántico.
* **Contenido:**
  * Pruebas unitarias visuales de la clase `CircuitFactory` instanciando circuitos aleatorios y algoritmos estructurados (QAOA, QFT, Grover).
  * Comparativa empírica entre la distribución analítica ideal (obtenida mediante precisión exacta de `Statevector`, actuando como *Ground Truth* $Y$) y la distribución ruidosa simulada con *shot noise* finito a 1024/4096 *shots* (Variable $X$).
  * Renderizado e inspección de los grafos resultantes procesados con `QuantumGraphExtractor`.
* **Valor para el Portfolio:** Demuestra el dominio de la API de Qiskit 1.0 y la rigurosidad estadística al aislar el ruido de muestreo para evitar el sobreajuste del modelo.

### 🤖 `03_gem_transformer_evaluation.ipynb` — Evaluación del Módulo GEM
* **Objetivo:** Analizar la convergencia y la precisión del Graph Transformer mitigando el ruido dinámico de las compuertas lógicas.
* **Contenido:**
  * Evaluación del rendimiento del regresor continuo sobre el conjunto de test *Out-Of-Distribution* (OOD) temporal (Días 8-10).
  * Curvas de aprendizaje, análisis de residuos matemáticos y distribución del error en la predicción del valor de desviación continuo ($\Delta$).
  * Pruebas de estrés variando la profundidad del circuito (de 5 a 50 capas) para identificar el punto de colapso del modelo.
* **Valor para el Portfolio:** Justifica la elección de la arquitectura basada en atención para capturar el paralelismo y la localidad cuántica sin violar los unitarios lógicos.

### 🕸️ `04_rem_gnn_matrix_free.ipynb` — Evaluación del Módulo REM y Subespacios
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
  * Análisis de compromiso (*Trade-off Analysis*) mediante gráficos de radar: **Ganancia de Fidelidad Cuántica vs. Latencia de Inferencia vs. Coste Computacional**.
  * Exportación automatizada de figuras y tablas en alta resolución para su inclusión directa en la memoria LaTeX del TFM.
* **Valor para el Portfolio:** Demuestra un enfoque de ingeniería de sistemas y MLOps maduro, orientado al rendimiento y la viabilidad del producto en entornos corporativos reales.

---

## 📐 Reglas de Oro para el Desarrollo en estos Notebooks

Para mantener la calidad de nivel de producción de este repositorio, el desarrollo dentro de este directorio debe regirse por los siguientes principios estrictos:

1. **Cero Funciones Kilométricas:** Si una celda requiere más de 20 líneas de manipulación de datos o tensores, esa lógica debe ser refactorizada, trasladada a `src/utils.py` o `src/dataset.py`, e importada en el notebook.
2. **Reproducibilidad Absoluta:** La primera celda de código de cada notebook debe inicializar las semillas de aleatoriedad global importadas desde `src/config.py`. Cualquier ejecución completa (*Run All*) debe replicar exactamente las mismas curvas y figuras.
3. **Documentación Narrativa Extensiva:** Cada sección de código debe estar precedida por bloques de Markdown detallados que expliquen el fundamento matemático (álgebra lineal, operadores de Kraus, matrices de densidad o subespacios lineales) de la operación. Las gráficas deben incluir etiquetas claras en sus ejes, leyendas y conclusiones analíticas escritas.
