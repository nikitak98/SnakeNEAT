import unittest

from vision import dxdy_eight, dxdy_four, look_direction


class TestVisionHelpers(unittest.TestCase):
    def test_dxdy_four_mappings(self):
        self.assertEqual(dxdy_four(0), (0, 1))
        self.assertEqual(dxdy_four(1), (-1, 0))
        self.assertEqual(dxdy_four(2), (0, -1))
        self.assertEqual(dxdy_four(3), (1, 0))

    def test_dxdy_eight_mappings(self):
        self.assertEqual(dxdy_eight(0), (0, 1))
        self.assertEqual(dxdy_eight(2), (-1, 0))
        self.assertEqual(dxdy_eight(4), (0, -1))
        self.assertEqual(dxdy_eight(6), (1, 0))

    def test_look_direction_detects_food_and_body(self):
        snake = [(20, 20), (0, 20), (80, 20)]
        food = (60, 20)

        distance_to_wall, distance_to_food, distance_to_body = look_direction(
            6, snake, food
        )

        self.assertAlmostEqual(distance_to_wall, 1 / 9)
        self.assertAlmostEqual(distance_to_food, 1 / 2)
        self.assertAlmostEqual(distance_to_body, 1 / 3)

    def test_look_direction_returns_zero_when_no_targets(self):
        snake = [(20, 20), (0, 20)]
        food = (200, 200)

        _, distance_to_food, distance_to_body = look_direction(0, snake, food)

        self.assertEqual(distance_to_food, 0)
        self.assertEqual(distance_to_body, 0)


if __name__ == "__main__":
    unittest.main()
