# Model Updating — review and recommendations: Digital Twin vs Digital Shadow

**Status:** recommendation, not yet implemented. Written 2026-10-07 after the rig
test session, for review by the team and the other agents working in this repo.
Line references are to the tree at that date, on branch `peshawa_codes`.

**Who should act on this:** whoever next owns the Model Updating tab. Items are
ordered by value and each carries its own acceptance criteria. Nothing here is
urgent for a rig test; item 1 is a correctness bug and should be done first.

---

## 1. The organising principle

This is the rule everything below follows, and the one to hold onto when deciding
where something belongs:

> **Digital Twin** = what the model says the structure *should* do, and what the
> calibration *implies about* the structure.
> **Digital Shadow** = where the real structure *departs from* the model, live.

A twin is fed the commanded input and predicts. A shadow runs alongside the real
thing and reports divergence. Anything that reads a calibration and tells you to
change the building is a **twin** output. Anything that compares a measured
signal against a running simulation is a **shadow** output.

---

## 2. What already changed (2026-10-07) — context, already done

So reviewers do not re-propose these:

1. **Tab layout.** Top level is now Live Signals, Spectrum, Model Updating,
   Sonification, Settings. "Digital Twin Experiment" was renamed **Digital
   Shadow** and moved *inside* Model Updating. Model Updating's sub-tabs are
   Model, Additional Mass, Analysis, Calibration, **Digital Twin**, **Digital
   Shadow**. The old **Output** sub-tab is what is now called Digital Twin.
2. **Nesting mechanism.** `MainWindow` still builds and owns the shadow
   (`MainWindow.digital_twin_tab`) and hands it over with
   `ModelUpdatingTab.host_digital_shadow()`. The shadow needs the Model Updating
   tab as a collaborator, so building it inside would be circular; adopting a
   finished widget keeps the dependency one-directional.
3. **Calibration defaults.** Analysis scope now opens on *Frequency + mode
   shapes*; mass target on *Total mass including additional masses*. Calibration
   method was already Bayesian (first in the list).
4. **The shadow's decision panel was removed.** It was a second rendering of what
   the Digital Twin sub-tab already owns (`ModelUpdatingTab._twin_panel`, fed per
   Continuous Update cycle). Decisions are a twin output; a shadow does not need
   them. The shadow keeps its comparison plots, the numerical 3D model and the
   live physical wireframe.

Pinned by `tests/test_tab_layout.py` (18 tests). **Do not rename the internals:**
the tab the UI calls *Digital Shadow* is `tab_digital_twin.py`, backed by the
`sensepi.digital_twin` package and writing to `output/digital_twin/`; the sub-tab
the UI calls *Digital Twin* is `ModelUpdatingTab._output_tab`. The names cross
over. Renaming the package or the output folder would move users' saved
experiment data, and is explicitly out of scope.

---

## 3. Findings

### FINDING 1 — the shadow's own Calibrate ignores the chosen calibration method *(bug)*

Model Updating's **Calibrate** honours the Calibration tab's method selector: it
branches on `calibration_method` and calls `run_bayesian_calibration` when that is
`"bayesian"` ([`tab_model_updating.py:708`](../src/sensepi/gui/tabs/tab_model_updating.py)).

The shadow's **"1. Calibrate"** does not. `_CalibrationWorker` imports
`run_calibration` and calls it unconditionally
([`tab_digital_twin.py:408`](../src/sensepi/gui/tabs/tab_digital_twin.py)) — the
least-squares calibrator. Its own docstring says so: *"identification, story
mapping, and the least-squares fit."*

So a user can set **Calibration method: Bayesian**, press the shadow's step 1, and
silently receive a least-squares fit. Since Bayesian is now the default
(change 3 above), **this mismatch is the default case.** The two copies of "the
same" calibration have drifted, which is what duplicated logic does.

This also undermines the Bayesian work: the confidence band and prior→posterior
behaviour described in the Continuous Update worker simply do not apply to a
shadow experiment calibrated in-tab.

### FINDING 2 — the shadow's Calibrate is a duplicate whose justification expired

`_CalibrationWorker` is `Identify & Analyze` minus the run, implemented a second
time. The code states the intent:

> *"Deliberately calls the pure `run_calibration` rather than going through the
> Model Updating tab — that is what lets this tab own its own calibration instead
> of borrowing one."*

That was sound **while the shadow was a separate top-level tab**: avoiding a
dependency on a sibling tab is reasonable. The shadow is now a sub-tab of the very
tab it was avoiding. The justification no longer holds, and the cost of the
duplication is FINDING 1.

### FINDING 3 — the calibration in force is computed and then never displayed *(root cause of the confusion)*

Four things can produce the calibration the shadow consumes:

| producer | where |
|---|---|
| Model Updating **Calibrate** | `_start_worker("calibrate")` |
| Model Updating **Identify & Analyze** | `_start_identify(chain=["calibrate","run"])` |
| **Use Latest Calibration for Digital Twin** | `_use_latest_continuous_for_digital_twin()` (`:3983`) |
| the shadow's own **1. Calibrate** | `_CalibrationWorker` |

with an implicit precedence in `_build_setup`
([`tab_digital_twin.py:~705`](../src/sensepi/gui/tabs/tab_digital_twin.py)): the
shadow's own wins, otherwise `_model_tab.calibration_snapshot()`, otherwise raise.

The code **computes which one is in force** and stores it in
`self._calibration_source` — and then **never renders it**. There are four
references: one init, two assignments, one entry into the setup dict. Zero
display.

The single fact that determines what the experiment is doing is not on screen
anywhere. This, not the number of buttons, is why the tab is confusing.

### FINDING 4 — nine preconditions, all discovered by pressing a button

`_calibrate` can refuse five ways: no sensor placement; no live capture source; no
sensor data yet (needs ~`sensor_window_s`, default 30 s); none of the streaming
sensors placed on a floor; identification failed (async). `_arm` can refuse four
more: Model Updating busy; no calibrated model; no live samples for the mapped
sensors; no sensor→story mapping.

Each is a `QMessageBox` found only by pressing the button. There is no standing
readiness display, so "what does it need?" can only be answered by failing.

### FINDING 5 — a button that should be a selection

**Use Latest Calibration for Digital Twin** exists only to copy
`_latest_continuous_calibration` into `_calibration_state` so another tab can see
it, and it **disables itself after one use** (`:4001`). That is manual data
plumbing exposed as a verb, and it makes the choice one-shot and irreversible
rather than visible and switchable.

### FINDING 6 — the step numbering describes three of five steps

"1. Calibrate / 2. Arm numerical model / 3. Start experiment" implies three
mandatory steps in order. In reality:

1. start the **live stream** (Arm is disabled without it; Calibrate fails without ~30 s of data)
2. produce a calibration — **optional here**, it may already exist from Model Updating
3. Arm
4. Start experiment
5. **start the physical shaker** (the model then waits for detected onset)

Two steps are missing from the numbering and the optional one is numbered as
mandatory.

### FINDING 7 — sensor buttons are gated from another sub-tab

`Load & Identify`, `Identify & Analyze`, `Start Continuous Update` and `Use Latest
Calibration for DT` are all disabled unless the **Calibration** sub-tab's
*Experimental data source* is `"From Sensors"` (`_on_exp_source_changed`, `:2592`).
A user looking at greyed-out buttons has no indication that the cause is a combo
box on a different sub-tab.

### FINDING 8 — the shadow is driven by the commanded input, not the measured one *(not cosmetic)*

`load_ground_motion` reads `params["gmFile"]`
([`opensees_runner.py:17`](../src/sensepi/digital_twin/opensees_runner.py)), and the
tab states it: *"No base sensor is used: this input is assumed to be the shaker
input."* The base sensor is used only for onset detection.

If the shaker does not faithfully reproduce its command, the shadow attributes the
difference to the structure. On a rig that already streams a base sensor, that is
a whole class of false divergence. This changes what the experiment measures, so
it deserves its own discussion rather than being folded into a UI cleanup.

### What is already right — do not "fix" these

* The `Lag:` spin box **is** correctly disabled while Auto-sync is on
  (`_on_auto_align_changed`, and again in `_update_controls`).
* Auto-sync deliberately forces the alignment lag to **0** and synchronises the
  *start instant* instead, which preserves genuine physical-vs-model phase
  difference instead of cross-correlating it away.
* The model definition is snapshotted at Calibrate, so edits in Model Updating
  cannot corrupt an experiment already under way.
* `Load & Identify` vs `Identify & Analyze` is **not** redundant: the former
  deliberately works without OpenSees and `_start_identify` redirects users to it
  when the extras are missing. Keep both.
* The onset trigger (0.5 s moving RMS, `max(6×background, 0.45×input)`, 0.25 s
  hold, base-sensor preference, catch-up offset) is well engineered and its
  thresholds are justified in comments from 170 recorded runs. Leave it alone.

---

## 4. Recommendations

### R1 — Make the shadow consume calibrations, never produce them *(fixes FINDING 1, 2; high value)*

Delete `_CalibrationWorker` from `tab_digital_twin.py`. If the convenience button
stays, have it delegate to Model Updating's calibration path so there is exactly
one implementation and one method selector.

This fixes the Bayesian bug *by construction* rather than by patching a second
branch, removes ~70 lines, and collapses the precedence rule in `_build_setup` —
with one producer there is nothing left to prioritise.

It also matches the organising principle: calibrating is a twin activity; the
shadow's job is to run a model it was given and report divergence.

**Acceptance criteria**
- With `Calibration method: Bayesian`, a shadow experiment runs on a Bayesian fit.
  A test must assert the method actually used, not just that calibration finished.
- `grep -c "run_calibration" src/sensepi/gui/tabs/tab_digital_twin.py` → 0.
- `_build_setup` has one calibration source; its "prefer this tab" branch is gone.
- Existing `tests/test_tab_layout.py` still passes.

*If the team decides the shadow must keep an independent calibration*, then the
minimum fix for FINDING 1 is to branch on `calibration_method` inside
`_CalibrationWorker` exactly as `tab_model_updating.py:708` does — but that keeps
two copies in sync by hand, which is how this drifted in the first place.

### R2 — Display the calibration in force *(fixes FINDING 3; cheapest high-value change)*

One line above the Arm button: source, when it was produced, the frequencies it
was fitted to, and whether it converged. `_calibration_source` already exists and
only needs rendering; `_CalibrationState.input_signature` and the saved
`saved_at` give the rest.

Show explicitly when there is **no** calibration, rather than leaving the user to
discover it by pressing Arm.

**Acceptance criteria**
- The shadow shows a non-empty calibration-source line in all three states: none,
  from Model Updating, from a Continuous Update handoff.
- A test asserts the label changes when the source changes.

### R3 — Replace the nine dialogs with a readiness panel *(fixes FINDING 4, 7)*

Four lines with ✓/✗ and the remedy for each:

| check | remedy shown |
|---|---|
| live stream running | "Start the stream in Live Signals" |
| sensor placement set | "Settings → Sensor placement map" |
| sensor→story mapping | "Model Updating → Model" |
| calibration available | "Calibrate, or Identify & Analyze" |

Keep Arm disabled until all four pass, with a tooltip naming the first failure.
The same panel should state when sensor buttons are disabled because
*Experimental data source* is not `"From Sensors"` (FINDING 7).

This converts nine modal failures into a standing answer to "what does it need?".

**Acceptance criteria**
- Each of the four checks can be driven false independently in a test and reports
  false.
- Arm is disabled whenever any check is false, enabled when all pass.
- No `QMessageBox` in the Arm path for a condition the panel already covers.

### R4 — Turn the handoff button into a source selector *(fixes FINDING 5)*

Replace **Use Latest Calibration for Digital Twin** with a combo in the shadow:
*Calibration to use: [ Model Updating's latest | latest Continuous Update cycle |
none ]*. Visible, reversible, no self-disabling button. Pairs naturally with R2,
which displays the resulting choice.

### R5 — Drop the misleading numbering *(fixes FINDING 6)*

Either number all five steps or drop the numbers and drive a state line instead:
`NOT READY → READY → ARMED → WAITING FOR SHAKER ONSET → RUNNING → FINISHED`. The
status label already carries most of this text; make the state explicit rather
than implied by prose.

### R6 — Rename buttons for what they consume and produce

| now | suggested |
|---|---|
| Load & Identify | **Identify from recording** (no OpenSees) |
| Identify & Analyze | **From sensors: identify → calibrate → run** |
| Use Latest Calibration for Digital Twin | *(removed by R4)* |

### R7 — Consider driving the shadow with the measured base acceleration *(FINDING 8; separate decision)*

Optional input source: *commanded (ground-motion file)* or *measured (base
sensor)*. The second removes shaker-fidelity error from the comparison. This
changes what the experiment measures and needs a decision from the structural
side, not just a UI change. Do not bundle it with R1–R6.

---

## 5. Suggested order

**R1 + R2 + R3 as one change.** They are three views of the same problem —
invisible, multiply-sourced state — and splitting them leaves the tab in a
half-explained condition. R1 is also a correctness fix, so it should not wait.

**Then R4, R5, R6** as cheap follow-up polish.

**R7 separately**, after the structural discussion.

---

## 6. Constraints for whoever implements this

* **Guardrails** (`.claude/CLAUDE.md` G1–G9) apply. In particular: calibration and
  identification stay on worker threads (G1/G4); nothing in
  `sensepi.analysis/core/sensors/config` may import from `sensepi.gui` (G7);
  openseespy stays optional in `pyproject.toml` and every use stays guarded (G8).
* **Do not rename** `tab_digital_twin.py`, the `sensepi.digital_twin` package, the
  `DigitalTwin*` classes, `MainWindow.digital_twin_tab`, or `output/digital_twin/`.
  The label/internal crossover is deliberate and documented; renaming the output
  folder would orphan saved experiment data.
* **Keep `MainWindow.digital_twin_tab`.** `closeEvent` and several tests use it.
* **Keep the nesting mechanism.** The shadow must be built after the Model
  Updating tab and adopted via `host_digital_shadow()`; constructing it inside
  Model Updating would be circular.
* **Run the full suite**: `QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests`.
  Baseline at the time of writing: **494 tests, 0 failures, 1 skip.** It is green —
  do not leave it otherwise. Note that several Qt defects in this repo only appear
  in a *full* run (a leaked widget outliving its children), so a passing single
  file is not sufficient evidence.
* **Add tests with each item.** The existing `tests/test_tab_layout.py` is the
  model to follow: assert the user-visible contract, and say in the docstring
  which defect the test exists to prevent.

---

## 7. Further reading in this repo

* `.claude/model_updating_twin_and_shadow.md` — how both sub-tabs work today:
  the shadow's three-step workflow, how its wall clock is tied to the real shaker
  (baseline, onset trigger, catch-up offset), why it cannot run without a
  calibration, and what every action button does. Read this before implementing.
* `src/sensepi/digital_twin/decisions.py` — why decisions are exactly one
  stiffness and one mass per storey, and why saturated bounds are reported as
  "at least".
* `src/sensepi/digital_twin/opensees_runner.py` — the real-time integration loop
  and the `a_abs = a_rel + a_ground` conversion.
* `.claude/issues.md`, `.claude/audit_2026_09.md` — existing debt and the earlier
  tab-by-tab audit.
