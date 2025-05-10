import sys
import os
import unittest
import numpy as np

# Ensure src folder is on PYTHONPATH for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from azul.rules import (
    validate_origin,
    place_on_pattern_line,
    transfer_to_wall,
    calculate_round_score,
    calculate_final_bonus,
    Color,
)

class TestRules(unittest.TestCase):

    def test_validate_origin_factory(self):
        factories = [[2, 0, 1, 0, 0], [0, 1, 0, 0, 0]]
        center = [0] * 5
        # color present in factory 0
        self.assertTrue(validate_origin(factories, center, ("factory", 0), Color.BLUE))
        # color absent in factory 1
        self.assertFalse(validate_origin(factories, center, ("factory", 1), Color.RED))

    def test_validate_origin_center(self):
        factories = [[0] * 5 for _ in range(2)]
        center = [0, 2, 0, 0, 0]
        self.assertTrue(validate_origin(factories, center, ("center", None), Color.YELLOW))
        self.assertFalse(validate_origin(factories, center, ("center", None), Color.BLUE))

    def test_place_on_pattern_line_empty(self):
        line = [-1, -1, -1]
        new_line, overflow = place_on_pattern_line(line, Color.RED, 2)
        expected_line = [Color.RED, Color.RED, -1]
        self.assertEqual(new_line, expected_line)
        self.assertEqual(overflow, 0)

    def test_place_on_pattern_line_mixed_color(self):
        # if line has another color, all overflow
        line = [Color.BLUE, -1]
        new_line, overflow = place_on_pattern_line(line, Color.RED, 2)
        self.assertEqual(new_line, line)
        self.assertEqual(overflow, 2)

    def test_transfer_to_wall_simple(self):
        wall = np.full((5,5), -1, dtype=int)
        # row 0 pattern_line length 1
        pattern_line = [Color.BLUE]
        score = transfer_to_wall(wall, pattern_line, 0)
        # should place at col 0 and score = 1
        self.assertEqual(score, 1)
        self.assertEqual(wall[0][0], Color.BLUE)

    def test_transfer_to_wall_adjacent(self):
        wall = np.full((5,5), -1, dtype=int)
        # place two horizontal neighbors at (1,1) and (1,3)
        wall[1][1] = Color.YELLOW
        wall[1][3] = Color.YELLOW
        # now pattern_line for row 1
        pattern_line = [Color.YELLOW, Color.YELLOW]
        score = transfer_to_wall(wall, pattern_line, 1)
        # base 1 + contiguous count 2 = 3
        self.assertEqual(score, 3)

    def test_calculate_round_score(self):
        # floor line with tiles in first, third, and sixth positions
        floor = [1, 0, 2, 0, 0, 3, 0]
        score = calculate_round_score(None, floor)
        # penalties: pos0=-1, pos2=-2, pos5=-3 => total -6
        self.assertEqual(score, -6)

    def test_calculate_final_bonus_full_wall(self):
        # A fully filled 5x5 wall with each row containing all colors
        wall = [
            [Color.BLUE, Color.YELLOW, Color.ORANGE, Color.BLACK, Color.RED]
            for _ in range(5)
        ]
        bonus = calculate_final_bonus(wall)
        # 5 complete rows *2 + 5 complete columns *7 + 5 complete color sets *10 = 85
        self.assertEqual(bonus, 2*5 + 7*5 + 10*5)

if __name__ == "__main__":
    unittest.main()