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
### 3. Orquestación y MLOps
* **`train.py`**: Pipeline de entrenamiento de la comparativa (alcance vigente jul-2026): entrena los 3 modelos — Ridge, Random Forest y GEM — con protocolo idéntico (mismos splits, métricas y semilla). Integra MLflow / Weights & Biases para el registro automatizado de métricas, artefactos y versionado de pesos.
* **`features.py` / `baselines.py`** (pendientes): features agregadas del circuito para los baselines tabulares, y los modelos Ridge + Random Forest (sklearn).
* **`inference.py`**: (reservado) punto de inferencia del GEM en producción; se definirá en TAREA 4.

### 4. Configuración y Utilidades Core
* **`config.py`**: El centro de control del repositorio. Ningún valor de hiperparámetro está *hardcodeado* en el código fuente. Este archivo gestiona las variables de entorno (`.env`), topes físicos (ej. 15 qubits max, 1024 shots) y, crucialmente, las semillas aleatorias (`seeds`) globales para asegurar el determinismo estadístico en toda la investigación.
* **`utils.py`**: Librería de funciones puras. Contiene operaciones matemáticas aisladas que ensucian la legibilidad de los modelos, como las métricas de la comparativa (MAE, RMSE, R², mejora relativa), *One-Hot Encodings* y utilidades numéricas compartidas.

---

## 📐 Principios de Desarrollo Soportados

1. **Desacoplamiento de Ciclo de Vida:** Las modificaciones en el simulador de ruido (`quantum_gen.py`) no rompen el código de carga de datos (`dataset.py`) ni alteran la arquitectura de la red (`gem_model.py`).
2. **Reproducibilidad:** Todas las llamadas estocásticas dependen de un estado centralizado en `config.py`.
3. **Escalabilidad de Memoria:** Prohibido construir objetos de tamaño $2^n \times 2^n$; el ground truth usa Statevector ($2^n$) acotado por `MAX_QUBITS=15` para ejecutar en 16 GB de RAM.
