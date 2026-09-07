"""Turn a calibration result into instructions for the PHYSICAL structure.

Every other part of this application runs the loop one way: measure the real
structure, then change the numerical model until it agrees. That produces a model
that matches reality, and stops there.

This module runs the loop the other way. If the calibrated model needs more mass
on storey 1 than the design specifies, then the real structure is behaving as
though it carries that extra mass — so the instruction is "take mass off storey
1", or equivalently "add it elsewhere to rebalance". The model stops being a
description and becomes a decision.

Why the decisions are exactly one stiffness and one mass per storey
-------------------------------------------------------------------
Not a choice made here — it is the shape of the optimiser's own variable vector
in ``opensees_model_updating.calibration.calibrator.run_calibration``::

    x[0]  = E_scale        one global stiffness scale
    x[1:] = mass_scale     one per storey

So these are a readout of what the calibration decided, not a reinterpretation of
it. That matters, because mass and stiffness are confounded in a modal fit: a
frequency shift can be explained by either. Attributing the whole discrepancy to
mass would be a guess. Reporting the split the calibrator itself chose is not —
though that split is still influenced by the bounds and by the mass
regularisation, which :func:`describe_confidence` states rather than hides.

Saturation
----------
Each scale is optimised inside bounds. A scale resting **on** its bound means the
fit wanted to go further and was not allowed to, so the true discrepancy is at
least what is reported. Those decisions are marked ``saturated`` and phrased as
"at least", because a number that looks exact when it is really a clipped bound
is worse than no number.

Pure and Qt-free (G7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Relative change below which nothing is worth doing. A calibration will never
#: return exactly 1.000, and chasing 1 % of a storey mass on a laboratory rig is
#: noise, not a finding.
DEFAULT_MASS_TOLERANCE = 0.05
DEFAULT_STIFFNESS_TOLERANCE = 0.05

#: How close to a bound counts as sitting on it.
BOUND_EPS = 1e-3

ADD = "add"
REMOVE = "remove"
NOTHING = "nothing"
STIFFEN = "stiffen"
SOFTEN = "soften"


@dataclass
class Decision:
    """One instruction about the physical structure."""

    target: str                  # "mass" | "stiffness"
    action: str                  # add | remove | nothing | stiffen | soften
    story: int = 0               # 1..n for mass; 0 for the global stiffness
    designed: float = 0.0
    calibrated: float = 0.0
    #: calibrated - designed, in the same units as the two above.
    delta: float = 0.0
    #: delta as a fraction of the designed value.
    relative: float = 0.0
    #: The fit hit its bound, so the real discrepancy is at least this much.
    saturated: bool = False

    @property
    def is_action(self) -> bool:
        return self.action != NOTHING

    def headline(self) -> str:
        """One short line, as it appears next to the light."""
        if self.action == NOTHING:
            return "do nothing"
        at_least = "at least " if self.saturated else ""
        if self.target == "stiffness":
            verb = "STIFFEN" if self.action == STIFFEN else "SOFTEN"
            return f"{verb} by {at_least}{abs(self.relative) * 100:.0f}%"
        verb = "ADD" if self.action == ADD else "REMOVE"
        return f"{verb} {at_least}{abs(self.delta):.3f}"

    def detail(self) -> str:
        """The reasoning, for the tooltip — never only the conclusion."""
        if self.target == "stiffness":
            what = "The calibrated model is"
            direction = ("softer" if self.action == STIFFEN else "stiffer")
            if self.action == NOTHING:
                return (f"Calibrated stiffness {self.calibrated:.4g} vs designed "
                        f"{self.designed:.4g} ({self.relative * 100:+.1f}%) — "
                        f"within tolerance.")
            note = (" The scale hit its bound, so the real difference is larger "
                    "than this." if self.saturated else "")
            return (f"{what} {direction} than the design "
                    f"({self.calibrated:.4g} vs {self.designed:.4g}, "
                    f"{self.relative * 100:+.1f}%), so the real structure is "
                    f"{direction} than intended.{note}")
        if self.action == NOTHING:
            return (f"Storey {self.story}: calibrated {self.calibrated:.4g} vs "
                    f"designed {self.designed:.4g} ({self.relative * 100:+.1f}%) "
                    f"— within tolerance.")
        heavier = self.delta > 0
        note = (" The scale hit its bound, so the real difference is larger than "
                "this." if self.saturated else "")
        return (
            f"Storey {self.story} behaves {'heavier' if heavier else 'lighter'} "
            f"than designed: the fit needed {self.calibrated:.4g} where the design "
            f"says {self.designed:.4g} ({self.relative * 100:+.1f}%). "
            f"{'Remove' if heavier else 'Add'} {abs(self.delta):.4g} to bring the "
            f"structure back to the design.{note}")


@dataclass
class DecisionSet:
    """Every instruction from one calibration, plus how far to trust them."""

    stiffness: Decision | None = None
    masses: list = field(default_factory=list)
    #: Rebalance alternative: move mass between storeys instead of changing the
    #: total. ``{story: delta}`` — positive means add there.
    rebalance: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    ok: bool = False
    message: str = ""

    @property
    def actions(self) -> list:
        out = [d for d in self.masses if d.is_action]
        if self.stiffness is not None and self.stiffness.is_action:
            out.append(self.stiffness)
        return out

    def summary(self) -> str:
        if not self.ok:
            return self.message or "No calibration yet."
        if not self.actions:
            return ("The structure matches its design within tolerance — "
                    "no change needed.")
        return f"{len(self.actions)} change(s) suggested to match the design."


def _saturated(value: float, lo: float, hi: float) -> bool:
    try:
        return bool(value <= float(lo) + BOUND_EPS or value >= float(hi) - BOUND_EPS)
    except (TypeError, ValueError):
        return False


def decide(
    designed: dict,
    calibrated: dict,
    *,
    mass_tolerance: float = DEFAULT_MASS_TOLERANCE,
    stiffness_tolerance: float = DEFAULT_STIFFNESS_TOLERANCE,
) -> DecisionSet:
    """Compare a calibrated model against the design it started from.

    ``designed`` and ``calibrated`` are the two parameter dicts the calibrator
    takes in and hands back — same keys, so a missing one means the calibration
    did not complete rather than that nothing changed.
    """
    out = DecisionSet()
    if not isinstance(designed, dict) or not isinstance(calibrated, dict):
        out.message = "No calibration result to compare against the design."
        return out

    d_masses = list(designed.get("floor_masses") or [])
    c_masses = list(calibrated.get("floor_masses") or [])
    if not d_masses or len(d_masses) != len(c_masses):
        out.message = ("The calibrated model and the design do not describe the "
                       "same number of storeys.")
        return out

    m_lb = designed.get("m_scale_lb", 0.0)
    m_ub = designed.get("m_scale_ub", 1e9)
    for i, (d, c) in enumerate(zip(d_masses, c_masses), start=1):
        d, c = float(d), float(c)
        delta = c - d
        rel = delta / d if d else 0.0
        scale = c / d if d else 1.0
        if abs(rel) <= mass_tolerance:
            action = NOTHING
        else:
            # The model needed MORE mass to reproduce the measurement, so the
            # real structure is carrying more than the design says: take it off.
            action = REMOVE if delta > 0 else ADD
        out.masses.append(Decision(
            target="mass", action=action, story=i, designed=d, calibrated=c,
            delta=delta, relative=rel,
            saturated=_saturated(scale, m_lb, m_ub) and action != NOTHING))

    d_e = float(designed.get("E", 0.0) or 0.0)
    c_e = float(calibrated.get("E", 0.0) or 0.0)
    if d_e > 0:
        rel = (c_e - d_e) / d_e
        if abs(rel) <= stiffness_tolerance:
            action = NOTHING
        else:
            # Calibrated stiffness BELOW the design means the real structure is
            # softer than intended, so it needs stiffening.
            action = STIFFEN if rel < 0 else SOFTEN
        out.stiffness = Decision(
            target="stiffness", action=action, story=0, designed=d_e,
            calibrated=c_e, delta=c_e - d_e, relative=rel,
            saturated=_saturated(c_e / d_e, designed.get("E_scale_lb", 0.0),
                                 designed.get("E_scale_ub", 1e9))
            and action != NOTHING)

    out.rebalance = rebalance_plan(out.masses)
    out.notes = describe_confidence(designed, out)
    out.ok = True
    return out


def rebalance_plan(masses: list) -> dict:
    """Move mass between storeys instead of changing the total.

    The user's alternative framing: rather than removing 0.16 from storey 1, add
    it to storey 2 so the *distribution* matches the design. Only meaningful when
    the total is roughly preserved, which is why the total imbalance is returned
    under key 0 — if that is large, redistribution alone cannot fix it.
    """
    plan = {d.story: -d.delta for d in masses if d.is_action}
    if plan:
        plan[0] = float(sum(plan.values()))
    return plan


def describe_confidence(designed: dict, decisions: DecisionSet) -> list:
    """What the reader needs in order to weigh these instructions.

    Stated on screen rather than buried, because each of these can change the
    conclusion, and a decision panel that shows only conclusions invites more
    trust than the numbers have earned.
    """
    notes = []
    notes.append(
        "Mass and stiffness are confounded in a modal fit: the same frequency "
        "shift can come from either. These are the split the calibrator chose, "
        "which the bounds and the mass regularisation influence.")
    if any(d.saturated for d in decisions.masses) or (
            decisions.stiffness is not None and decisions.stiffness.saturated):
        notes.append(
            "At least one scale is resting on its bound — the fit wanted to go "
            "further and could not, so those differences are lower limits.")
    if not designed.get("use_mode_shapes", True):
        notes.append(
            "Calibration used frequencies only. Frequencies alone constrain the "
            "total, not where it sits, so per-storey attribution is weak.")
    return notes
