import unittest


class OfficeSvgCompatibilityTest(unittest.TestCase):
    def test_embedded_icons_use_xlink_compatible_references(self):
        from MetaBCIWebGame.export_structure4_svg import build_svg

        svg = build_svg()

        self.assertIn('xmlns:xlink="http://www.w3.org/1999/xlink"', svg)
        self.assertIn('xlink:href="data:image/png;base64,', svg)

    def test_four_class_control_uses_square_vector_direction_icon(self):
        from MetaBCIWebGame.export_structure4_svg import build_svg

        svg = build_svg()

        self.assertEqual(svg.count('id="direction-four-class"'), 1)

    def test_illustrator_version_does_not_use_referenced_arrow_markers(self):
        from MetaBCIWebGame.export_structure4_svg import build_svg

        svg = build_svg()

        self.assertNotIn("<marker", svg)
        self.assertNotIn("marker-end=", svg)

    def test_growing_window_sample_count_is_wrapped_inside_card(self):
        from MetaBCIWebGame.export_structure4_svg import build_svg

        svg = build_svg()

        self.assertNotIn("125 / 250 / 375 / 500 采样点", svg)
        self.assertIn(">125 / 250 / 375 / 500</text>", svg)
        self.assertIn(">采样点</text>", svg)


if __name__ == "__main__":
    unittest.main()
