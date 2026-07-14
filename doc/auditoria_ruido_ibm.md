# Auditoría: nuestro modelo de ruido vs. la documentación oficial de IBM/Qiskit Aer

> Encargo (jul-2026): revisar la documentación de la plataforma cuántica de IBM y de Qiskit Aer
> sobre inyección de ruido, y contrastarla con la implementación de `src/quantum_gen.py`
> (`build_noise_model_for_day()` / `_populate_noise_model()`). **Este documento SOLO analiza —
> no se ha modificado código.** Toda suposición está marcada [SUPOSICIÓN].

## 0. Fuentes consultadas

| Fuente | Qué cubre |
|---|---|
| [Aer — Noise Models (API)](https://qiskit.github.io/qiskit-aer/apidocs/aer_noise.html) | `NoiseModel`, `depolarizing_error`, `ReadoutError`, `thermal_relaxation_error` |
| [Aer — Building Noise Models (tutorial)](https://qiskit.github.io/qiskit-aer/tutorials/3_building_noise_models.html) | Construcción manual de modelos |
| [Aer — Device noise simulation (tutorial)](https://qiskit.github.io/qiskit-aer/tutorials/2_device_noise_simulation.html) | El camino canónico `NoiseModel.from_backend()` |
| [IBM Quantum — Build noise models (guía)](https://quantum.cloud.ibm.com/docs/en/guides/build-noise-models) | Guía oficial de la plataforma |
| [Issue Aer #1655](https://github.com/Qiskit/qiskit-aer/issues/1655) | Caveat conocido de `from_backend` |
| Verificación empírica propia contra `ibm_kingston` real (14-jul-2026) | Campos disponibles en `backend.properties()` |

## 1. Qué dice la documentación (el modelo canónico)

El camino documentado para simular un dispositivo real es **`NoiseModel.from_backend(backend)`**, que construye automáticamente:

1. **Error de puerta 1q** = despolarización **seguida de** relajación térmica (T1/T2 con la duración real `gate_length` de CADA puerta). La probabilidad de despolarización se **calibra para que la infidelidad media combinada** (depol + térmica) **iguale exactamente el `gate_error` medido** del backend.
2. **Error de puerta 2q** = despolarización 2q + relajación térmica en cada qubit participante — usando el **`gate_error` real de ESA puerta 2q concreta** (no una heurística).
3. **Readout** desde las probabilidades medidas del dispositivo.
4. Toggles independientes: `gate_error=`, `thermal_relaxation=`, `readout_error=` — el desacople de fuentes de ruido que nuestro diseño original quería está soportado de serie.

## 2. Qué hace nuestro `quantum_gen.py`

Modelo **manual**: (a) `depolarizing_error(p=gate_error_sx, 1)` en puertas 1q; (b) `depolarizing_error(p=5×media(gate_error_sx), 2)` en CZ [SUPOSICIÓN declarada en código]; (c) `ReadoutError` **simétrico** desde `readout_error`; (d) **sin** relajación térmica (retirada por el bug "Kraus is empty" al combinarla a mano con depolarizing en TAREA 3); (e) drift step-like sintético multiplicando las propiedades por día.

## 3. Hallazgos (ordenados por severidad)

### H1 — 🔴 Probabilidad de despolarización ≠ gate_error: inyectamos ~la MITAD del error en 1q
La doc de `from_backend` es explícita: la probabilidad de despolarización se calibra para que la **infidelidad** iguale al `gate_error`. Nosotros usamos directamente `p = gate_error`, pero para un canal despolarizante la infidelidad media es menor que `p`:
- [SUPOSICIÓN — derivación propia, verificar con `qiskit.quantum_info.average_gate_fidelity` antes de tocar código] infidelidad ≈ `p·(d−1)/d` → 1q (d=2): `p/2`; 2q (d=4): `3p/4`.
- Consecuencia: con `p = gate_error` estamos inyectando **≈50% del error 1q documentado** (y ~75% en 2q respecto a la p elegida). Los Δ del dataset están sistemáticamente **infra-estimados** en magnitud.
- Atenuante: afecta por igual a los 3 modelos de la comparativa (el ranking no se invalida), pero sí a cualquier lectura física absoluta de Δ.

### H2 — 🔴 La heurística "CZ ≈ 5×" es innecesaria: el error CZ real por par ESTÁ disponible
Verificado empíricamente contra `ibm_kingston` (14-jul-2026): `props.gate_error("cz", [u, v])` devuelve el error medido de cada par físico. Muestras reales: `cz(0,1)=9.2e-4` (ratio 3.5× vs sx), `cz(1,2)=1.2e-3` (8.2×), `cz(2,3)=1.5e-3` (6.6×).
- Nuestra media uniforme 5× cae dentro del rango real — la [SUPOSICIÓN] no era mala — pero **pierde la variación espacial por par**, que es exactamente la señal que la literatura pre-ejecución explota (mapomatic, Hartnett et al.).
- Con calibración real ya descargada, no hay razón para mantener la heurística en el full run.

### H3 — 🟠 La relajación térmica falta, y el camino canónico la incluye sin nuestro bug
`from_backend` combina depol+térmica de fábrica sin el error "Kraus is empty" — nuestro fallo de TAREA 3 fue la combinación MANUAL con parámetros incompatibles [SUPOSICIÓN: la composición calibrada que hace from_backend evita el caso degenerado; no reproduje el bug para confirmarlo]. Efecto físico ausente en nuestro modelo: la dependencia del error con la DURACIÓN del circuito (circuitos largos sufren más decoherencia por tiempo, no solo por número de puertas).

### H4 — 🟠 Readout simétrico vs. realidad asimétrica 2.4×
Verificado en `ibm_kingston` q0: `prob_meas0_prep1 = 0.0117` vs `prob_meas1_prep0 = 0.0049`. Los campos que TAREA 7a quería confirmar **existen con esos nombres exactos** en `qubit_property()`. Nuestro `ReadoutError` simétrico usa la media (0.0083) — pierde una asimetría real de 2.4×.

### H5 — 🟡 Reimplementamos parcialmente lo que `from_backend` da en una línea
Nuestro modelo manual existía para poder inyectar el drift sintético por día. Pero ahora que tenemos **29 snapshots de calibración real** (jun–jul 2026, `data/raw/calib_history/`), existe un camino mejor: [SUPOSICIÓN — verificar que existe en Aer 0.17] `NoiseModel.from_backend_properties(props_del_día)` permitiría construir el modelo canónico completo (depol calibrada + térmica + readout, todo real por puerta) **para cada día histórico** — eliminando de golpe H1+H2+H3 y sustituyendo el drift sintético por drift real.

### H6 — 🟢 Cosas que SÍ están alineadas con la documentación
- `depolarizing_error(p, n)` bien invocada en firma y rango; `ReadoutError([[1-p,p],[p,1-p]])` formato correcto.
- Registrar el error CZ en ambas orientaciones del par: correcto (Aer asocia el error al orden exacto de qubits).
- `AerSimulator(method='statevector')` con noise model → trayectorias estocásticas por shot: es el comportamiento documentado; `seed_simulator` garantiza reproducibilidad.
- `NoiseModel(basis_gates=...)` coherente con las puertas nativas Heron.
- El aviso del caché de calibración y el patrón de descarga son consistentes con la guía de la plataforma.

### H7 — ⚪ Caveat de la comunidad
Existe un issue conocido ([Aer #1655](https://github.com/Qiskit/qiskit-aer/issues/1655)) sobre `from_backend` y la conversión de gate_error en ciertos casos — si se migra a `from_backend`, conviene validar la infidelidad resultante con un circuito de referencia antes de generar el dataset completo.

## 4. Recomendaciones (NO aplicadas — pendientes de tu decisión)

| # | Acción | Resuelve | Esfuerzo |
|---|---|---|---|
| R1 | Migrar el noise model a `from_backend_properties(snapshot_del_día)` usando los 29 días reales | H1+H2+H3+H5 y el drift pasa a ser 100% real | Medio (reescribir `build_noise_model_for_day` + regenerar dataset) |
| R2 | (Alternativa mínima) mantener el modelo manual pero: usar CZ real por par y corregir p↔infidelidad | H1+H2 | Bajo |
| R3 | Readout asimétrico con `prob_meas1_prep0`/`prob_meas0_prep1` (TAREA 7a, ahora trivial) | H4 | Bajo |
| R4 | Validación: comparar la infidelidad efectiva del modelo elegido contra un circuito de referencia | H7 | Bajo |

**Mi recomendación: R1** — un solo cambio convierte el punto más cuestionable del TFM (ruido sintético con suposiciones) en "modelo de ruido canónico de la documentación de IBM, alimentado con 29 días de calibración real". Es, además, la respuesta perfecta a la pregunta del tutor sobre "datos reales de IBM".
