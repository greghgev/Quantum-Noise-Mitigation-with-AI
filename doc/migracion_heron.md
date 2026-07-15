# Migración Eagle → Heron (julio 2026)

> Registro de la migración forzosa del backend de referencia del TFM, sus consecuencias
> técnicas y las limitaciones que introduce. Complementa a `ROADMAP.md` (TAREA 3b) y
> `fallas_y_soluciones.md`. Leer antes de redactar §6 (limitaciones) de la memoria.

---

## 1. Motivo

IBM retiró **toda la familia Eagle** entre 2023 y abril de 2026:

| Máquina | Retirada |
|---|---|
| ibm_washington | jun 2023 |
| ibm_osaka, ibm_cusco | ago 2024 |
| ibm_kyoto | sep 2024 |
| ibm_nazca | dic 2024 |
| **ibm_kyiv** (backend original del TFM) | **18 abr 2025** |
| ibm_sherbrooke | jul 2025 |
| ibm_brisbane | nov 2025 |
| ibm_brussels, ibm_strasbourg (últimos Eagle) | 27 abr 2026 |

Fuente: [Retired QPUs — IBM Quantum Documentation](https://quantum.cloud.ibm.com/docs/en/guides/retired-qpus).
Cita textual de IBM: *"The global fleet has now fully transitioned away from Eagle"*.

**Backend nuevo:** `ibm_kingston` — Heron r2, 156 qubits, accesible en Open Plan.

## 2. Diferencias técnicas Eagle vs. Heron relevantes para el TFM

| Aspecto | Eagle r3 (ibm_kyiv) | Heron r2 (ibm_kingston) |
|---|---|---|
| Qubits | 127 | 156 |
| Puerta 2q nativa | **CX** (direccional) | **CZ** (simétrica) |
| Puertas 1q | id, rz, sx, x | id, rz, sx, x (sin cambio) |
| Topología | Heavy-Hex | Variante heavy-hex, 176 acopladores sintonizables |
| Error 2q | ~2–5×10⁻³ | ~5× menor (EPLG@100q ≈ 2.15×10⁻³ en r3) |
| Velocidad puertas | baseline | ~8× más rápidas |
| Crosstalk ZZ residual | significativo | suprimido por acopladores sintonizables |

## 3. Cambios realizados en el proyecto

- `src/config.py`: nuevas constantes `BACKEND_NAME` (lee `IBM_BACKEND_NAME` de `.env`,
  default `ibm_kingston`) y `BASIS_GATES = ["cz", "id", "rz", "sx", "x"]`.
- `src/quantum_gen.py`: transpilación a base CZ; one-hot posición 0 = `cz`; noise model
  registra el error CZ **en ambas orientaciones** del par (CZ es simétrica pero Aer asocia
  el error al orden exacto de qubits del circuito transpilado).
- `scripts/make_synthetic_calib.py`: calibración sintética Heron r2 reproducible mientras no
  hubo credenciales (eliminado el 14-jul-2026 al llegar la calibración real — historial git).
- `tests/test_quantum_gen.py`: adaptados a CZ — 36/36 PASSED.
- Mini-dataset Eagle (178 muestras) eliminado y regenerado con base Heron.
- Documentación actualizada: CLAUDE.md, README.md, ROADMAP.md, dataset_info.md,
  IDEA_GENERAL_TFM.md, IDEAS_FUTURAS.md, notebooks_info.md.
- `doc/SOTA/comparative_analysis.md` **no** se modifica: sus menciones a ibm_kyiv son
  hechos históricos de los papers (QEM-Bench se evaluó en ibm_kyiv), no configuración nuestra.

## 4. Bloqueado por credenciales IBM (recuperable)

✅ RESUELTO (14-jul-2026): IBM aprobó la verificación; upgrade completado, instancia Open Plan
activa y credenciales en `.env`. Los puntos 1–4 de abajo quedaron ejecutados ese mismo día
(calibración real descargada — con fix del `EdgeList` no serializable, tal y como predecía el
punto 5 histórico —, dataset regenerado, e histórico de 29 días vía `collect_calibration.py --backfill`).

1. ✅ **Calibración real** — descargada de `ibm_kingston` (156q, 352 aristas); la sintética fue eliminada.
2. ✅ **Coupling map real** — el transpilador usa ya la topología heavy-hex real (la cadena lineal 20q era del apaño sintético).
3. ⏳ **Error CZ real por par** — la heurística "CZ ≈ 5× error 1q" SIGUE ACTIVA [SUPOSICIÓN]: la
   calibración cachea solo gate_error de `sx`. Sustituirla por el error CZ medido por arista es la
   recomendación R1 de `doc/auditoria_ruido_ibm.md` (pendiente de decisión).
4. ✅ **gate_length real** — ya viene de properties() reales.
5. ✅ **Campo `prob_meas1_prep0`** — CONFIRMADO en la API real (readout asimétrico disponible para TAREA 7a).
6. 🔄 **TAREA 6** — desbloqueada: 6a (calibración diaria) EN MARCHA con 29 días backfilled;
   6b (ejecución en hardware) pendiente, cuota Open Plan 10 min/mes disponible.

### Pasos ejecutados al recuperar credenciales (14-jul-2026, registro)

Los pasos previstos se ejecutaron tal cual: credenciales a `.env` → borrado de calibración
sintética → descarga real vía `HardwareTelemetrics` (con fix del `EdgeList` de rustworkx no
serializable a JSON) → `pytest` 37/37 → regeneración del mini-dataset → `dvc add && dvc push`.
Además: `scripts/collect_calibration.py --backfill 30` archivó 29 snapshots históricos reales.

## 5. Pérdidas estructurales (no recuperables — documentar en §6 de la memoria)

1. **Comparabilidad directa con QEM-Bench rota.** QEMFormer reporta MAE=0.018/RMSE=0.026
   *en ibm_kyiv* (Eagle). Con Heron (~5× menos error de puerta) los Δ son sistemáticamente
   menores y un MAE más bajo no significa mejor modelo. *Mitigación:* comparar solo métricas
   relativas (R², ratio de mejora, mejora relativa vs. no mitigar) y declararlo. Nota: nadie
   puede ya reproducir QEM-Bench en su hardware original — limitación de todo el campo.
2. **Evidencia de drift pre-Heron.** Los saltos step-like ±15–30% (Hirasaki et al., APL 2023)
   se midieron en dispositivos anteriores. [SUPOSICIÓN] Que Heron drifte igual es plausible
   (misma física de qubits superconductores) pero no está publicado.
3. **Semántica control/target degradada.** Con CX, las posiciones [9:12] (control) y [14:17]
   (target) del feature vector capturaban una asimetría física real. CZ es simétrica: la
   distinción pasa a ser el orden arbitrario del transpilador (control=qargs[0]). El GEM ya
   no puede aprender asimetrías control/target porque no existen físicamente.
4. **Mini-dataset Eagle descartado** (178 muestras). Recuperable del remote DVC vía historial
   git si se quisiera un estudio comparativo Eagle-vs-Heron sintético.
5. **Fractional gates no explotadas.** Heron soporta RZZ(θ) nativa opcional que implementaría
   cada término ZZ de QAOA/TFIM en una sola puerta. No se usa: rompería el one-hot de 6 tipos
   y la homogeneidad con la literatura. Anotado como extensión en IDEAS_FUTURAS.

## 6. Efecto colateral positivo

**El caso para TAREA 7b (crosstalk) se debilita.** Los acopladores sintonizables de Heron
existen precisamente para suprimir el acoplamiento ZZ residual. Si la brecha
simulation-to-real de TAREA 6 sale pequeña, TAREA 7b puede cerrarse como "innecesaria en
Heron" con respaldo bibliográfico, ahorrando esfuerzo de implementación.

## 7. Registro de suposiciones activas post-migración

| # | Suposición | Ubicación | ¿Verificable con credenciales? |
|---|---|---|---|
| S1 | ~~Calibración sintética~~ | — | ✅ RESUELTA (real desde 14-jul-2026) |
| S2 | ~~Cadena lineal 20q~~ | — | ✅ RESUELTA (coupling map real, 352 aristas) |
| S3 | Error CZ ≈ 5× error 1q (posible infraestimación; ¿~10×?) | `_populate_noise_model()` | ⏳ ACTIVA — resolver con R1 de la auditoría |
| S4 | Drift step-like ±15–30% aplica a Heron | `_init_drift_schedule()` | ⚠️ Requeriría telemetría histórica propia |
| S5 | Readout simétrico P(0→1)=P(1→0) | `_populate_noise_model()` | ⏳ ACTIVA — campos asimétricos confirmados en API (TAREA 7a) |
| S6 | Métricas relativas comparables Eagle (SOTA) vs Heron (TFM) | memoria §6 | ❌ Argumentar, no verificar |
