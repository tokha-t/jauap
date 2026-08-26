from __future__ import annotations

import unittest

from jauap.geo import resolve_location


class RuralResolutionTests(unittest.TestCase):
    def test_kyzyl_zhuldyz_resolves_to_committed_settlement_centroid(self) -> None:
        location = resolve_location(
            "Кызыл-Жулдыз ауылында дорога разбита және мусор шығарылмайды",
            use_llm=False,
        )

        self.assertIsNotNone(location)
        assert location is not None
        self.assertEqual(location.settlement, "Кызыл-Жулдыз")
        self.assertEqual((location.lat, location.lon), (53.365, 69.285))


if __name__ == "__main__":
    unittest.main()
