## Análisis Arquitectónico y Soluciones de Diseño (MLOps)

Durante el diseño del pipeline se identificaron cuellos de botella críticos. Aquí se documentan los que afectan al alcance vigente del TFM (GEM + comparativa de modelos); el análisis del módulo de corrección de lectura del diseño original se consolidó íntegro en `IDEAS_FUTURAS.md` (IDEA-0).

### 1. La "Trampa del Transpilador" (Módulo GEM)

* **El Problema:** Inicialmente se planteó que el Graph Transformer (GEM) recomendara "reordenamientos o cancelaciones" de puertas lógicas en el circuito (DAG). Sin embargo, un circuito cuántico representa una matriz unitaria exacta $\mathcal{U}$. Si un modelo estocástico elimina o altera puertas arbitrariamente para evitar el ruido, destruye la integridad lógica del algoritmo del usuario, cambiando el resultado matemático esperado.
* **La Solución (Predicción de Valor Esperado):** El Transformer no actúa como un compilador que reescribe el circuito, sino como un "tasador de ruido". El modelo ingiere el circuito original y la calibración del chip, y predice un valor continuo de desviación ($\Delta$). El circuito original se ejecuta intacto en el hardware real, obteniendo un valor ruidoso $\langle O \rangle_{\text{ruido}}$. Finalmente, aplicamos una corrección algebraica en el post-procesado: $\langle O \rangle_{\text{mitigado}} = \langle O \rangle_{\text{ruido}} - \Delta$. De esta forma, mitigamos el ruido sin violar la matemática del circuito.

---

### 2. La estrategia de shots (etiquetas vs. histogramas)

Dado que vas a generar tu dataset de entrenamiento en tu ordenador usando el simulador qiskit-aer, nosotros controlamos las reglas del juego. **Actualización (jul-2026, tras detectar el problema en el EDA de la migración Heron — ver `doc/migracion_heron.md` y `ROADMAP.md` TAREA 8): hay que distinguir dos usos de "shots" que este documento originalmente mezclaba.**

- **Shots para HISTOGRAMAS realistas (`TRAIN_SHOTS` en `config.py`, actualmente 1024, reservado para usos futuros):** cuando un modelo consuma distribuciones medidas, debe aprender a sobrevivir en el "barro" — entrenar con 100.000 shots sobreajustaría a distribuciones casi perfectas que nunca aparecen en hardware real.

- **Shots para ESTIMAR LA ETIQUETA Δ del GEM (`LABEL_SHOTS` en `config.py`, actualmente 4096):** aquí el razonamiento es el CONTRARIO. Δ es ground truth — cuanto más ruido de muestreo (shot noise) lleve la etiqueta, peor aprende el modelo, sin ningún beneficio de "realismo". [SUPOSICIÓN] Con 4096 shots el suelo de shot-noise es ≈±0.005 — del mismo orden que el propio Δ medido en circuitos poco profundos (QAOA reps=1, QFT), es decir la etiqueta es en esos casos indistinguible del ruido de medición. Pendiente subir a 8192/16384 antes del full run (TAREA 8, ROADMAP.md).

- Para la variable "Ground Truth" del **valor esperado ideal**: Qiskit nos permite extraer el vector de estado matemático ideal (probabilidad exacta e infinita, sin shot noise) vía `Statevector`. Se usa siempre, para calcular la mitad "ideal" de Δ — nunca tiene el problema anterior.

---
