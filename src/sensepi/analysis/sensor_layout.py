"""Turn the Settings sensor map into the inputs the analysis engine accepts.

This module sits **above** :mod:`sensepi.analysis.modal` and does not change it.
That separation is deliberate: ``identify_modes``, ``map_to_stories`` and
``to_experimental_dict`` are shared with the Model Updating tab (and, through it,
with Digital Twin). Changing their signature or behaviour would push placement
semantics into those tabs before they have been reviewed. So the map-aware logic
lives here, each tab adopts it when its turn comes, and the shared engine keeps
behaving exactly as it did.

Pure and Qt-free (G7): it consumes the plain dict produced by
``SensorMapWidget.current_map().to_mapping()``, never the widget.

What the map decides
--------------------
* **which sensors are responses** — floor >= 1. The floor-0 sensor measures the
  shaker *input*, so it is not a structural DOF and must not enter an
  output-only identification as if it were one.
* **which channel** — the excitation axis picks ``ax`` or ``ay``. One at a time:
  the rig is shaken along a single horizontal direction and the structure
  responds along it. ``az`` is vertical and carries no lateral mode-shape or
  torsion information, so it is not offered. ``gz`` is the yaw rate, used for
  torsion rather than for mode shapes.
* **how many modes can be claimed** — you can only identify as many modes as you
  have independent measurement points. A 5-storey frame has 5 modes; 4 sensors
  can resolve at most 4 of them, and their *shapes* only over the floors those
  sensors actually cover. The cap is applied here so no tab can ask the engine
  for a shape it cannot support.
* **where torsion is available** — a floor carrying two sensors in different plan
  cells yields a differenced torsion indicator. It is *relative*: no plan
  dimension is recorded anywhere in the configuration, so an absolute rotation
  cannot be computed and is not claimed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CENTRE_CELL = "B2"

#: Excitation axis -> the accelerometer channel that measures the response
#: along it. ``az`` is intentionally absent (vertical; see module docstring).
AXIS_CHANNEL = {"x": "ax", "y": "ay"}

#: Channel used for the direct torsion (yaw-rate) reading.
TORSION_CHANNEL = "gz"

DEFAULT_MODES = 3


@dataclass(frozen=True)
class AnalysisLayout:
    """Everything the spectrum/modal path needs, derived from the sensor map."""

    n_floors: int = 3
    axis: str = "x"
    #: ``{sensor_id: floor}`` for structural sensors only (floor >= 1).
    story_map: dict = field(default_factory=dict)
    #: ``{sensor_id: plan cell}`` for structural sensors only.
    cell_map: dict = field(default_factory=dict)
    base_sensor_id: int | None = None
    #: ``{floor: (sensor_a, sensor_b)}`` — floors carrying a differenceable pair.
    torsion_pairs: dict = field(default_factory=dict)
    preset: str = ""
    warnings: list = field(default_factory=list)

    # -- derived ----------------------------------------------------------
    @property
    def channel(self) -> str:
        """The accelerometer channel for the selected excitation axis."""
        return AXIS_CHANNEL.get(self.axis, "ax")

    @property
    def structural_ids(self) -> list:
        return sorted(self.story_map)

    @property
    def covered_floors(self) -> list:
        return sorted(set(self.story_map.values()))

    @property
    def missing_floors(self) -> list:
        covered = set(self.story_map.values())
        return [f for f in range(1, self.n_floors + 1) if f not in covered]

    @property
    def has_base(self) -> bool:
        return self.base_sensor_id is not None

    @property
    def has_torsion(self) -> bool:
        return bool(self.torsion_pairs)

    @property
    def is_valid(self) -> bool:
        """At least one structural sensor — otherwise there is nothing to identify."""
        return bool(self.story_map)

    def max_modes(self, requested: int = DEFAULT_MODES) -> int:
        """Modes the sensor count can support. Never fewer than 1."""
        return max(1, min(int(requested), len(self.story_map) or 1))

    def max_shape_modes(self, requested: int = DEFAULT_MODES) -> int:
        """Modes whose *shape* the floor coverage can support.

        Sensors sharing a floor are averaged into one story value, so the shape
        resolution is set by distinct covered floors, not by sensor count.
        """
        return max(1, min(int(requested), len(self.covered_floors) or 1))

    def story_map_list(self, sensor_ids) -> list:
        """Adapter to ``map_to_stories``' positional convention.

        That function takes a **list** indexed by position in the data array, not
        a ``{sensor_id: floor}`` dict. ``0`` means "not on a story", which it
        drops. Sensors absent from the map — including the base sensor — map to
        ``0`` and so never enter a story.
        """
        return [int(self.story_map.get(int(sid), 0)) for sid in sensor_ids]

    def response_rows(self, sensor_ids) -> list:
        """Indices into a per-sensor data array that are structural responses.

        Used to drop the base row before an output-only identification: the base
        measures the input, and feeding it in as a response biases the picked
        modes toward the excitation.
        """
        return [i for i, sid in enumerate(sensor_ids)
                if int(sid) in self.story_map]

    def base_row(self, sensor_ids):
        """Index of the base sensor in a per-sensor data array, or ``None``.

        ``None`` covers both "no base sensor is placed" and "one is placed but is
        not among the streaming sensors" — the caller must handle the second, or
        base-referenced identification would silently fall back to output-only.
        """
        if self.base_sensor_id is None:
            return None
        for i, sid in enumerate(sensor_ids):
            if int(sid) == int(self.base_sensor_id):
                return i
        return None

    def describe(self) -> str:
        """One-line human summary — what Spectrum shows instead of its own picker."""
        if not self.is_valid:
            return ("No structural sensor is placed. Set the placement in "
                    "Settings → Sensor placement map.")
        floors = ", ".join(str(f) for f in self.covered_floors)
        bits = [f"{self.n_floors} floors",
                f"{len(self.story_map)} sensor(s) on floor(s) {floors}",
                f"shaking along {self.axis.upper()} → {self.channel}"]
        bits.append("base sensor S%d (input)" % self.base_sensor_id
                    if self.has_base else "no base sensor")
        if self.has_torsion:
            bits.append("torsion pair on floor(s) "
                        + ", ".join(str(f) for f in sorted(self.torsion_pairs)))
        return " · ".join(bits)


def layout_from_mapping(mapping, requested_modes: int = DEFAULT_MODES) -> AnalysisLayout:
    """Build an :class:`AnalysisLayout` from ``SensorMap.to_mapping()``.

    Tolerant by design: a missing, empty or malformed mapping yields a layout
    with no structural sensors, which every caller can detect via ``is_valid``
    and report rather than crash on.
    """
    if not isinstance(mapping, dict):
        return AnalysisLayout(warnings=["No sensor map available."])

    try:
        n_floors = max(1, int(mapping.get("n_floors") or 3))
    except (TypeError, ValueError):
        n_floors = 3
    axis = str(mapping.get("axis") or "x").lower()
    if axis not in AXIS_CHANNEL:
        axis = "x"
    preset = str(mapping.get("preset") or "")

    story_map: dict = {}
    cell_map: dict = {}
    base_id: int | None = None
    warnings: list = []

    for row in mapping.get("placements") or []:
        if not isinstance(row, dict):
            continue
        try:
            sid = int(row["sensor_id"])
        except (KeyError, TypeError, ValueError):
            continue
        floor = row.get("floor")
        if floor is None:
            continue
        try:
            floor = int(floor)
        except (TypeError, ValueError):
            continue
        if floor == 0:
            if base_id is None:
                base_id = sid
            else:
                warnings.append(
                    f"More than one sensor is on floor 0; S{base_id} is used as "
                    f"the base and S{sid} is ignored.")
            continue
        if not 1 <= floor <= n_floors:
            warnings.append(
                f"S{sid} is on floor {floor}, outside 1..{n_floors} — excluded.")
            continue
        story_map[sid] = floor
        cell_map[sid] = str(row.get("cell") or CENTRE_CELL)

    # Torsion: a floor with two sensors in DIFFERENT plan cells can be
    # differenced. Two sensors in the SAME cell measure the same point, so their
    # difference is noise, not rotation.
    by_floor: dict = {}
    for sid, floor in story_map.items():
        by_floor.setdefault(floor, []).append(sid)
    torsion_pairs: dict = {}
    for floor, ids in by_floor.items():
        ids = sorted(ids)
        if len(ids) < 2:
            continue
        cells = {sid: cell_map.get(sid, CENTRE_CELL) for sid in ids}
        first = ids[0]
        other = next((sid for sid in ids[1:] if cells[sid] != cells[first]), None)
        if other is None:
            warnings.append(
                f"Floor {floor} carries {len(ids)} sensors in the same plan cell "
                f"— averaged, no torsion.")
            continue
        torsion_pairs[floor] = (first, other)

    if not story_map:
        warnings.append("No structural sensor is placed.")
    if story_map and len(story_map) < requested_modes:
        warnings.append(
            f"{len(story_map)} structural sensor(s) can resolve at most "
            f"{len(story_map)} mode(s); {requested_modes} were requested.")

    missing = [f for f in range(1, n_floors + 1)
               if f not in set(story_map.values())]
    if missing and story_map:
        warnings.append(
            "Unmeasured floor(s) " + ", ".join(map(str, missing))
            + " — mode shapes omit them (never interpolated).")

    return AnalysisLayout(
        n_floors=n_floors, axis=axis, story_map=story_map, cell_map=cell_map,
        base_sensor_id=base_id, torsion_pairs=torsion_pairs, preset=preset,
        warnings=warnings)


def default_damping_sensor(layout: AnalysisLayout) -> int:
    """Sensor to read the free-decay from: the highest structural floor.

    The top of the structure has the largest first-mode amplitude, so it gives
    the cleanest decay envelope. Falls back to sensor 1 when nothing is placed.
    """
    if not layout.story_map:
        return 1
    top = max(layout.story_map.values())
    return min(sid for sid, floor in layout.story_map.items() if floor == top)
