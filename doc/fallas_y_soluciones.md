## Análisis Arquitectónico y Soluciones de Diseño (MLOps)

Durante el diseño del pipeline, se identificaron tres cuellos de botella críticos que comprometían la escalabilidad y la viabilidad matemática del modelo en producción. A continuación se detallan estas fallas y las soluciones implementadas en la arquitectura final:

### 1. La "Trampa del Transpilador" (Módulo GEM)

* **El Problema:** Inicialmente se planteó que el Graph Transformer (GEM) recomendara "reordenamientos o cancelaciones" de puertas lógicas en el circuito (DAG). Sin embargo, un circuito cuántico representa una matriz unitaria exacta $\mathcal{U}$. Si un modelo estocástico elimina o altera puertas arbitrariamente para evitar el ruido, destruye la integridad lógica del algoritmo del usuario, cambiando el resultado matemático esperado.
* **La Solución (Predicción de Valor Esperado):** El Transformer no actúa como un compilador que reescribe el circuito, sino como un "tasador de ruido". El modelo ingiere el circuito original y la calibración del chip, y predice un valor continuo de desviación ($\Delta$). El circuito original se ejecuta intacto en el hardware real, obteniendo un valor ruidoso $\langle O \rangle_{\text{ruido}}$. Finalmente, aplicamos una corrección algebraica en el post-procesado: $\langle O \rangle_{\text{mitigado}} = \langle O \rangle_{\text{ruido}} - \Delta$. De esta forma, mitigamos el ruido sin violar la matemática del circuito.

---

### 2. El Cuello de Botella Exponencial (Módulo REM)

* **El Problema:** La mitigación de errores de lectura clásica implica calcular e invertir una matriz de confusión global. Para un circuito de $n$ qubits, esta matriz tiene un tamaño de $2^n \times 2^n$. Su inversión requiere una complejidad de $\mathcal{O}(2^{3n})$. Al intentar escalar el modelo más allá de los 15-20 qubits (estándar en la era NISQ), el entorno colapsa por falta de memoria RAM (Out of Memory), invalidando el uso del pipeline para hardware real.
* **La Solución (Enfoque "Matrix-Free"):** Se implementa una solución de subespacios. La GNN no predice una matriz global inmanejable, sino que aprende las matrices de error locales marginales (por nodo/qubit) aprovechando la topología dispersa del chip. En el momento de la inferencia matemática, como el número de resultados distintos (bitstrings) está limitado físicamente por el número de *shots* (ej. 1024), formulamos un sistema de ecuaciones lineales $Ax = b$ reducido exclusivamente a los estados medidos. Esto reduce la complejidad a $\mathcal{O}(N)$ y permite escalar a más de 100 qubits sin impacto en memoria.

**Explicación más al detalle:**

El enfoque que usaremos se llama **"Matrix-Free Subspace Inversion"** (Inversión en Subespacios Libre de Matrices), fuertemente inspirado en algoritmos de la industria como el *Matrix-free Measurement Mitigation* (M3).

Aquí tienes el paso a paso matemático y computacional:

#### 1º. El Cambio de Paradigma: De Global a Local (El trabajo de tu GNN)

Si tienes un chip de 50 qubits, la matriz de error de lectura global ($A$) es de $2^{50} \times 2^{50}$. No cabe en ningún disco duro del planeta.

* **Lo que NO vas a hacer:** Tu IA no va a intentar predecir esa matriz gigante.
* **Lo que SÍ vas a hacer:** El error de los sensores es mayoritariamente local. Tu GNN va a predecir **matrices de confusión individuales de $2 \times 2$** para cada qubit (y si hay mucho *crosstalk*, matrices de $4 \times 4$ para qubits vecinos).
* Matemáticamente, la matriz global $A$ es el producto tensorial de estas pequeñas matrices locales:

$$A = A_1 \otimes A_2 \otimes \dots \otimes A_n$$


* **La regla de oro:** *Jamás* calcularemos ese producto tensorial en el código. Solo guardamos las matrices locales en memoria.

#### 2º. El Truco Físico: El Subespacio de los "Shots"

Aquí está la magia de la ingeniería de datos. En cuántica, el vector de probabilidad ideal ($x$) y el ruidoso ($b$) tienen longitud $2^n$.

* Pero cuando ejecutas un circuito en IBM, tú le dices a la máquina: *"Ejecuta esto 1024 veces (shots)"*.
* Esto significa que, como máximo, la máquina te va a devolver **1024 cadenas de bits (bitstrings) únicas**. Los otros $2^{50} - 1024$ estados posibles tienen cero ocurrencias. Son irrelevantes.
* Por lo tanto, no necesitamos resolver un sistema gigantesco. Solo necesitamos proyectar nuestro problema sobre el **subespacio** de los estados que realmente hemos observado.

#### 3º. La Ejecución "Matrix-Free" (Cómo se programa)

El objetivo es encontrar la distribución ideal ($x$) resolviendo el sistema lineal:


$$Ax = b$$


Donde $b$ son las mediciones ruidosas (tus *counts* de IBM) y $A$ es la matriz de error (que no tenemos construida).

* **El Algoritmo Iterativo:** En lugar de usar la inversa clásica ($x = A^{-1}b$), usaremos un solucionador iterativo de Krylov disponible en `scipy.sparse.linalg` (como **GMRES** o **BiCGSTAB**).
* **¿Por qué esto lo cambia todo?** GMRES no necesita conocer la matriz $A$. Solo necesita una función en Python que, dado un vector $v$, le devuelva el resultado de la multiplicación $A \cdot v$.
* **Tu Código:** Escribirás una función rápida que aplique secuencialmente las matrices locales de $2 \times 2$ (las que predijo tu GNN) sobre los bitstrings observados. GMRES usará esta función unas pocas decenas de veces y te devolverá el vector $x$ corregido en cuestión de milisegundos.

**NOTA: La Estrategia para tu TFM ->**

Dado que vas a generar tu dataset de entrenamiento en tu ordenador usando el simulador qiskit-aer, nosotros controlamos las reglas del juego. **Actualización (jul-2026, tras detectar el problema en el EDA de la migración Heron — ver `doc/migracion_heron.md` y `ROADMAP.md` TAREA 8): hay que distinguir dos usos de "shots" que este documento originalmente mezclaba.**

- **Shots para el INPUT del REM (`TRAIN_SHOTS` en `config.py`, actualmente 1024):** aquí sí aplica el razonamiento original — queremos que la IA aprenda a sobrevivir en el "barro". Entrenar con 100.000 shots sobreajustaría a distribuciones casi perfectas que nunca aparecen en hardware real.

- **Shots para ESTIMAR LA ETIQUETA Δ del GEM (`LABEL_SHOTS` en `config.py`, actualmente 4096):** aquí el razonamiento es el CONTRARIO. Δ es ground truth — cuanto más ruido de muestreo (shot noise) lleve la etiqueta, peor aprende el modelo, sin ningún beneficio de "realismo". [SUPOSICIÓN] Con 4096 shots el suelo de shot-noise es ≈±0.005 — del mismo orden que el propio Δ medido en circuitos poco profundos (QAOA reps=1, QFT), es decir la etiqueta es en esos casos indistinguible del ruido de medición. Pendiente subir a 8192/16384 antes del full run (TAREA 8, ROADMAP.md).

- Para la variable "Ground Truth" del **valor esperado ideal**: Qiskit nos permite extraer el vector de estado matemático ideal (probabilidad exacta e infinita, sin shot noise) vía `Statevector`. Se usa siempre, para calcular la mitad "ideal" de Δ — nunca tiene el problema anterior.

---

### 3. Acoplamiento Innecesario y Cuello de Botella MLOps

* **El Problema:** El diseño preliminar forzaba un flujo secuencial estricto donde la GNN del REM dependía de la salida del Transformer del GEM. Esto es un error a nivel MLOps, ya que impide el entrenamiento paralelo de los modelos. Además, a nivel físico, el ruido térmico y de lectura al final del circuito es estocásticamente independiente del ruido de las puertas operadas milisegundos antes.
* **La Solución (Desacoplamiento de Training vs. Inference):**
    * **Fase de Entrenamiento (Paralela):** Los modelos se entrenan en pipelines aislados. El GEM se entrena con circuitos profundos sujetos exclusivamente a ruido térmico y de despolarización (ignorando el readout). El REM se entrena con circuitos triviales (preparación y medida) sujetos exclusivamente a matrices de error de lectura.
    * **Fase de Inferencia (Secuencial):** El acoplamiento solo ocurre en producción. El usuario lanza su tarea, el GEM estima la desviación del valor esperado, el circuito se ejecuta en el QPU de IBM, y el REM limpia la distribución discreta de probabilidad resultante.

**Explicación más al detalle:**

#### ¿Cómo se hace el Entrenamiento en Paralelo? (El Truco del Gemelo Digital)

Gracias a que estás usando el simulador `qiskit-aer` para generar tus datos, tú eres "Dios" en ese gemelo digital. Puedes encender y apagar el ruido a placer.

* **Equipo A (Entrenando el REM - GNN):**
    * Apagas el ruido de las puertas en el simulador.
    * Generas miles de circuitos ridículamente simples: poner un qubit en 1 y medirlo. Ponerlo en 0 y medirlo.
    * Enciendes **solo el ruido de lectura**.
    * La GNN aprende exclusivamente cómo se equivoca el sensor (la matriz de confusión) sin que le moleste la complejidad de las puertas lógicas.


* **Equipo B (Entrenando el GEM - Transformer):**
    * Apagas el ruido de lectura en el simulador (asumes que el sensor es perfecto).
    * Generas circuitos profundos y complejos — **Random, HEA, Trotterized TFIM** (⚠️ corrección jul-2026: la versión original de esta nota decía "QAOA, QFT" como ejemplo, pero desde TAREA 1 ambos están reservados EXCLUSIVAMENTE para evaluación zero-shot y NUNCA aparecen en el pool de entrenamiento — ver `_CIRCUIT_TYPES_BY_SPLIT` en `quantum_gen.py` y `ROADMAP.md`).
    * Enciendes **solo el ruido térmico y de despolarización** de las puertas.
    * El Transformer aprende a predecir la degradación del circuito sin que el ruido de los sensores ensucie sus cálculos.



Como ves, puedes tener un script de Python en tu ordenador entrenando el GEM, y a la vez en Google Colab entrenando el REM. Son datasets distintos y modelos distintos.

### ¿Cómo se unen en Inferencia (Producción)?

Aquí es donde tu intuición acertaba. El día que defiendas el TFM y lances tu prueba final en el hardware real de IBM:

1. Lanzas el circuito real en IBM.
2. IBM te devuelve unos resultados (counts) sucios por todos lados.
3. Pisas esos datos primero con tu modelo **REM (GNN)**, que limpiará el error de los sensores.
4. Coges ese resultado parcialmente limpio, le restas el valor de desviación que predijo tu **GEM (Transformer)** para las puertas, y obtienes tu resultado mitigado final.

**En resumen:** En MLOps, divides y vencerás en el entrenamiento, y acoplas en la inferencia.
