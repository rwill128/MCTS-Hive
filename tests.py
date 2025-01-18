from TicTacToe.TicTacToe import TicTacToeGame

def run_tests():
    game = TicTacToeGame()

    try:
        # 1. Test X wins (top row)
        state_x_win = {
            "board": ["X", "X", "X", None, None, None, None, None, None],
            "current_player": 'O'
        }
        assert game.isTerminal(state_x_win), "State should be terminal when X has a row."
        assert game.getReward(state_x_win, 'X') == 1.0, "X should get reward 1.0 if X wins."
        assert game.getReward(state_x_win, 'O') == -1.0, "O should get reward -1.0 if X wins."
    except AssertionError as e:
        print(f"Test failed: X wins (top row): {e}")

    try:
        # 2. Test O wins (middle row)
        state_o_win = {
            "board": [None, None, None, "O", "O", "O", None, None, None],
            "current_player": 'X'
        }
        assert game.isTerminal(state_o_win), "State should be terminal when O has a row."
        assert game.getReward(state_o_win, 'O') == 1.0, "O should get reward 1.0 if O wins."
        assert game.getReward(state_o_win, 'X') == -1.0, "X should get reward -1.0 if O wins."
    except AssertionError as e:
        print(f"Test failed: O wins (middle row): {e}")

    try:
        # 3. Test Draw
        state_draw = {
            "board": ["X", "O", "X", "O", "O", "X", "X", "X", "O"],
            "current_player": 'X'
        }
        assert game.isTerminal(state_draw), "Full board with no winner is terminal."
        assert game.getReward(state_draw, 'X') == 0, "Draw should give X reward of 0."
        assert game.getReward(state_draw, 'O') == 0, "Draw should give O reward of 0."
    except AssertionError as e:
        print(f"Test failed: Draw scenario: {e}")

    try:
        # 4. Test a non-terminal state
        state_nonterminal = {
            "board": ["X", "O", None, None, "X", None, None, None, "O"],
            "current_player": 'O'
        }
        assert not game.isTerminal(state_nonterminal), "This is not a winning or full board."
        assert game.getReward(state_nonterminal, 'X') == 0.0, "No winner yet, reward should be 0 for partial state."
        assert game.getReward(state_nonterminal, 'O') == 0.0, "No winner yet, reward should be 0."
    except AssertionError as e:
        print(f"Test failed: Non-terminal state: {e}")

    print("Basic TicTacToeGame tests completed.")

def test_mcts_forced_win():
    game = TicTacToeGame()
    from mcts.Mcts import MCTS

    try:
        # State where X can win immediately by placing in cell 2.
        test_state = {
            "board": ["X", "X", None, "O", "O", None, None, None, None],
            "current_player": "X"
        }
        mcts = MCTS(game, num_iterations=5000, c_param=3)
        action = mcts.search(test_state)
        assert action == 2, f"MCTS should pick cell 2 to win immediately, but got {action}!"
    except AssertionError as e:
        print(f"Test failed: MCTS forced win for X: {e}")

    print("Test passed: MCTS found the forced winning move for X.")

def test_mcts_forced_block():
    game = TicTacToeGame()
    from mcts.Mcts import MCTS

    try:
        # O must block at cell 3 to prevent X from winning on the next move.
        test_state = {
            "board": [None, None, None, None, 'X', 'X', None, None, 'O'],
            "current_player": "O"
        }
        mcts = MCTS(game, num_iterations=5000, c_param=3)
        action = mcts.search(test_state)
        assert action == 3, f"MCTS should block X at cell 3. Got {action} instead."
    except AssertionError as e:
        print(f"Test failed: MCTS forced block for O: {e}")

    print("Test passed: MCTS found the forced block for O.")

def test_mcts_forced_block_x():
    game = TicTacToeGame()
    from mcts.Mcts import MCTS

    try:
        # X must block at cell 3 to prevent O from winning on the next move.
        test_state = {
            "board": [None, None, None, None, 'O', 'O', None, None, 'X'],
            "current_player": "X"
        }
        mcts = MCTS(game, num_iterations=5000, c_param=3)
        action = mcts.search(test_state)
        assert action == 3, f"MCTS should block O at cell 3. Got {action} instead."
    except AssertionError as e:
        print(f"Test failed: MCTS forced block for X: {e}")

    print("Test passed: MCTS found the forced block for X.")

if __name__ == "__main__":
    # run_tests()
    test_mcts_forced_win()
    test_mcts_forced_block()
    test_mcts_forced_block_x()
