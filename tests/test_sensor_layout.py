"""The Settings sensor map -> analysis inputs layer.

These tests pin the decisions that make the map *mean* something, rather than
just checking the dataclass round-trips:

* the floor-0 sensor is an INPUT and must never be treated as a response
* modes are capped by measurement points, not by storey count
* torsion needs two sensors in DIFFERENT plan cells
* the excitation axis chooses the channel, one at a time
"""
import unittest

from sensepi.analysis import sensor_layout as sl


def _map(n_floors, axis, rows, preset=""):
    return {"n_floors": n_floors, "axis": axis, "preset": preset,
            "placements": [{"sensor_id": s, "floor": f, "cell": c}
                           for s, f, c in rows]}


class TestBaseSensor(unittest.TestCase):
    def test_base_is_not_a_response(self):
        lay = sl.layout_from_mapping(_map(3, "x", [
            (1, 0, "B2"), (2, 1, "B2"), (3, 2, "B2"), (4, 3, "B2")]))
        self.assertEqual(lay.base_sensor_id, 1)
        self.assertNotIn(1, lay.story_map)
        # ...and its row is dropped from the identification data.
        self.assertEqual(lay.response_rows([1, 2, 3, 4]), [1, 2, 3])

    def test_story_map_list_zeroes_the_base(self):
        lay = sl.layout_from_mapping(_map(3, "x", [
            (1, 0, "B2"), (2, 1, "B2"), (3, 2, "B2"), (4, 3, "B2")]))
        # map_to_stories drops story 0, so the base contributes to no floor.
        self.assertEqual(lay.story_map_list([1, 2, 3, 4]), [0, 1, 2, 3])

    def test_second_floor_zero_sensor_is_reported_not_silently_used(self):
        lay = sl.layout_from_mapping(_map(3, "x", [
            (1, 0, "B2"), (2, 0, "A1"), (3, 1, "B2")]))
        self.assertEqual(lay.base_sensor_id, 1)
        self.assertTrue(any("S2" in w for w in lay.warnings))


class TestModeCap(unittest.TestCase):
    def test_five_storeys_four_sensors_cannot_claim_five_modes(self):
        lay = sl.layout_from_mapping(_map(5, "x", [
            (1, 1, "B2"), (2, 2, "B2"), (3, 4, "B2"), (4, 5, "B2")]))
        self.assertEqual(lay.n_floors, 5)
        self.assertEqual(lay.max_modes(5), 4)
        self.assertEqual(lay.max_modes(3), 3)

    def test_base_sensor_does_not_count_as_a_measurement_point(self):
        lay = sl.layout_from_mapping(_map(3, "x", [
            (1, 0, "B2"), (2, 1, "B2"), (3, 2, "B2"), (4, 3, "B2")]))
        self.assertEqual(lay.max_modes(4), 3)

    def test_shape_resolution_follows_distinct_floors_not_sensor_count(self):
        # Two sensors share the top floor: 3 sensors, but only 2 storeys.
        lay = sl.layout_from_mapping(_map(2, "x", [
            (1, 1, "B2"), (2, 2, "A1"), (3, 2, "C3")]))
        self.assertEqual(lay.max_modes(3), 3)
        self.assertEqual(lay.max_shape_modes(3), 2)

    def test_never_returns_zero(self):
        self.assertEqual(sl.layout_from_mapping(None).max_modes(3), 1)


class TestTorsion(unittest.TestCase):
    def test_pair_in_different_cells_gives_torsion(self):
        lay = sl.layout_from_mapping(_map(3, "x", [
            (1, 1, "B2"), (2, 2, "B2"), (3, 3, "A1"), (4, 3, "C3")]))
        self.assertTrue(lay.has_torsion)
        self.assertEqual(lay.torsion_pairs, {3: (3, 4)})

    def test_pair_in_the_same_cell_gives_none(self):
        lay = sl.layout_from_mapping(_map(3, "x", [
            (1, 3, "B2"), (2, 3, "B2")]))
        self.assertFalse(lay.has_torsion)
        self.assertTrue(any("same plan cell" in w for w in lay.warnings))

    def test_one_sensor_per_floor_gives_none(self):
        lay = sl.layout_from_mapping(_map(3, "x", [
            (1, 1, "B2"), (2, 2, "B2"), (3, 3, "B2")]))
        self.assertFalse(lay.has_torsion)


class TestAxis(unittest.TestCase):
    def test_axis_picks_one_channel(self):
        self.assertEqual(sl.layout_from_mapping(_map(1, "x", [(1, 1, "B2")])).channel, "ax")
        self.assertEqual(sl.layout_from_mapping(_map(1, "y", [(1, 1, "B2")])).channel, "ay")

    def test_vertical_is_never_offered(self):
        self.assertNotIn("az", sl.AXIS_CHANNEL.values())

    def test_unknown_axis_falls_back_to_x(self):
        self.assertEqual(sl.layout_from_mapping(_map(1, "z", [(1, 1, "B2")])).axis, "x")


class TestRobustness(unittest.TestCase):
    def test_no_map_is_reported_not_crashed(self):
        for bad in (None, {}, {"placements": None}, {"placements": [1, 2]}, "nope"):
            lay = sl.layout_from_mapping(bad)
            self.assertFalse(lay.is_valid)
            self.assertIn("Settings", lay.describe())

    def test_floor_above_the_building_is_excluded_and_reported(self):
        lay = sl.layout_from_mapping(_map(2, "x", [(1, 1, "B2"), (2, 7, "B2")]))
        self.assertEqual(lay.story_map, {1: 1})
        self.assertTrue(any("outside 1..2" in w for w in lay.warnings))

    def test_unmounted_sensor_is_simply_absent(self):
        lay = sl.layout_from_mapping({"n_floors": 2, "axis": "x", "placements": [
            {"sensor_id": 1, "floor": 1, "cell": "B2"},
            {"sensor_id": 2, "floor": None, "cell": "B2"}]})
        self.assertEqual(lay.story_map, {1: 1})
        self.assertEqual(lay.response_rows([1, 2]), [0])

    def test_unmeasured_floors_are_warned_about(self):
        lay = sl.layout_from_mapping(_map(5, "x", [(1, 1, "B2"), (2, 5, "B2")]))
        self.assertEqual(lay.missing_floors, [2, 3, 4])
        self.assertTrue(any("never interpolated" in w for w in lay.warnings))


class TestDampingSensor(unittest.TestCase):
    def test_defaults_to_the_top_floor(self):
        lay = sl.layout_from_mapping(_map(3, "x", [
            (1, 0, "B2"), (2, 1, "B2"), (3, 2, "B2"), (4, 3, "B2")]))
        self.assertEqual(sl.default_damping_sensor(lay), 4)

    def test_never_picks_the_base(self):
        lay = sl.layout_from_mapping(_map(1, "x", [(1, 0, "B2"), (2, 1, "B2")]))
        self.assertEqual(sl.default_damping_sensor(lay), 2)


if __name__ == "__main__":
    unittest.main()
