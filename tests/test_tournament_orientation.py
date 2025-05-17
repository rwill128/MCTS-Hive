import unittest

class TestTournamentOrientation(unittest.TestCase):
    def test_c4_key_generation(self):
        names = ['a', 'b']
        orientation = 1
        if orientation == 0:
            x_name, o_name = names[0], names[1]
        else:
            x_name, o_name = names[1], names[0]
        key = f"{x_name}_vs_{o_name}"
        self.assertEqual(key, 'b_vs_a')

    def test_ttt_key_generation(self):
        names = ['a', 'b']
        orientation = 1
        if orientation == 0:
            x_name, o_name = names[0], names[1]
        else:
            x_name, o_name = names[1], names[0]
        key = f"{x_name}_vs_{o_name}"
        self.assertEqual(key, 'b_vs_a')

if __name__ == '__main__':
    unittest.main()
