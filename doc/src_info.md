# Documentación del Código Fuente (`src/`)

Este directorio contiene la lógica de negocio y el núcleo algorítmico del proyecto **TFM-Quantum**. El código está diseñado bajo principios SOLID, programación orientada a objetos (POO) y estándares estrictos de MLOps.

El objetivo de esta estructura es garantizar la modularidad, la reproducibilidad y el desacoplamiento absoluto entre las fases de generación de datos, entrenamiento de modelos e inferencia en producción.

---

## 🏗️ Arquitectura de Módulos

### 1. Data Engineering & Pipeline de Datos
* **`quantum_gen.py`**: Es el motor de generación de circuitos sintéticos. Actúa como el Gemelo Digital del hardware de IBM. Contiene clases como `CircuitFactory` (generación matemática), `HardwareTelemetrics` (inyección de ruido de calibración) y `QuantumGraphExtractor` (traducción de circuitos cuánticos a grafos de PyTorch). Genera y exporta los archivos `.pt`.
* **`dataset.py`**: Interfaz de consumo para el entrenamiento. Contiene las clases que heredan de `torch_geometric.data.Dataset`. Se encarga de cargar los archivos `.pt` generados previamente en disco y empaquetarlos en `DataLoaders` eficientes para alimentar las GPUs/CPUs en lotes (batches).

### 2. Arquitecturas de Deep Learning (PyTorch)
Para evitar el acoplamiento y facilitar la depuración, los modelos residen en archivos separados:
* **`gem_model.py` (Gate Error Mitigation)**: Define la arquitectura del Graph Transformer. Ingiere el grafo del circuito y la telemetría de las compuertas lógicas para actuar como un regresor continuo, prediciendo la desviación ($\Delta$) del valor esperado debido al ruido térmico y de despolarización.
* **`rem_model.py` (Readout Error Mitigation)**: Define la arquitectura de la Graph Neural Network (GNN). Predice las matrices de confusión marginales locales para el error de los sensores láser. Incluye la lógica algorítmica para el enfoque *Matrix-Free*, proyectando la solución en el subespacio de *shots* observados para mantener una complejidad $\mathcal{O}(N)$.

### 3. Orquestación y MLOps
* **`train.py`**: Pipeline de entrenamiento desacoplado. Ejecuta los bucles de entrenamiento (`epochs`), optimización de gradientes y cálculo de la *Loss Function* (Fidelidad). Integra MLflow / Weights & Biases para el registro automatizado de métricas, artefactos y versionado de pesos de red. Entrena GEM y REM en procesos aislados para evitar fugas de datos (*Data Leakage*).
* **`inference.py`**: El pipeline de producción real. Es el único punto donde GEM y REM interactúan. Ingiere un circuito de usuario, ejecuta la predicción del GEM, simula la ejecución en el hardware real, aplica la mitigación del REM sobre el histograma ruidoso, y computa el ajuste algebraico final.

### 4. Configuración y Utilidades Core
* **`config.py`**: El centro de control del repositorio. Ningún valor de hiperparámetro está *hardcodeado* en el código fuente. Este archivo gestiona las variables de entorno (`.env`), topes físicos (ej. 15 qubits max, 1024 shots) y, crucialmente, las semillas aleatorias (`seeds`) globales para asegurar el determinismo estadístico en toda la investigación.
* **`utils.py`**: Librería de funciones puras. Contiene operaciones matemáticas aisladas que ensucian la legibilidad de los modelos, como el cálculo de tensores unitarios, *One-Hot Encodings*, cálculos de Fidelidad Cuántica y llamadas de bajo nivel al *solver* iterativo GMRES de SciPy.

---

## 📐 Principios de Desarrollo Soportados

1. **Desacoplamiento de Ciclo de Vida:** Las modificaciones en el simulador de ruido (`quantum_gen.py`) no rompen el código de carga de datos (`dataset.py`) ni alteran la arquitectura de la red (`gem_model.py`).
2. **Reproducibilidad:** Todas las llamadas estocásticas dependen de un estado centralizado en `config.py`.
3. **Escalabilidad de Memoria:** El código prohíbe explícitamente la construcción de matrices $2^n \times 2^n$ durante las fases de mitigación en `rem_model.py`, permitiendo su ejecución local en estaciones de trabajo con memoria RAM limitada.
