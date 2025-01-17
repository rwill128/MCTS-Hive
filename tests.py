from TicTacToe.TicTacToe import TicTacToeGame


def run_tests():
    game = TicTacToeGame()

    # 1. Test X wins (top row)
    # Board layout:
    #   X | X | X
    #  ---+---+---
    #     |   |
    #  ---+---+---
    #     |   |
    state_x_win = {
        "board": ["X", "X", "X", None, None, None, None, None, None],
        "current_player": 'O'  # next player doesn't matter here
    }
    assert game.isTerminal(state_x_win), "State should be terminal when X has a row."
    assert game.getReward(state_x_win, 'X') == 1.0, "X should get reward 1.0 if X wins."
    assert game.getReward(state_x_win, 'O') == 0.0, "O should get reward 0.0 if X wins."

    # 2. Test O wins (middle row)
    # Board layout:
    #     |
    #  ---+---+---
    #   O | O | O
    #  ---+---+---
    #     |
    state_o_win = {
        "board": [None, None, None, "O", "O", "O", None, None, None],
        "current_player": 'X'
    }
    assert game.isTerminal(state_o_win), "State should be terminal when O has a row."
    assert game.getReward(state_o_win, 'O') == 1.0, "O should get 1.0 if O wins."
    assert game.getReward(state_o_win, 'X') == 0.0, "X should get 0.0 if O wins."

    # 3. Test Draw
    # One possible full-board draw:
    #   X | O | X
    #  ---+---+---
    #   O | O | X
    #  ---+---+---
    #   X | X | O
    state_draw = {
        "board": ["X", "O", "X",
                  "O", "O", "X",
                  "X", "X", "O"],
        "current_player": 'X'  # next player doesn't matter
    }
    # No winner, board is full
    assert game.isTerminal(state_draw), "Full board with no winner is terminal."
    assert game.getReward(state_draw, 'X') == 0.5, "Draw should give X reward of 0.5."
    assert game.getReward(state_draw, 'O') == 0.5, "Draw should give O reward of 0.5."

    # 4. Test a non-terminal state
    # Board layout:
    #   X | O |
    #  ---+---+---
    #     | X |
    #  ---+---+---
    #     |   | O
    state_nonterminal = {
        "board": ["X", "O", None,
                  None, "X", None,
                  None, None, "O"],
        "current_player": 'O'  # Let's say O is to move
    }
    assert not game.isTerminal(state_nonterminal), "This is not a winning or full board."
    assert game.getReward(state_nonterminal, 'X') == 0.0, "No winner yet, reward should be 0 for partial state."
    assert game.getReward(state_nonterminal, 'O') == 0.0, "No winner yet, reward should be 0."

    print("All TicTacToeGame tests passed successfully!")


def test_mcts_forced_win():
    game = TicTacToeGame()
    from mcts.Mcts import MCTS  # your MCTS code

    # State where X can win immediately by placing in cell 2.
    test_state = {
        "board": ["X", "X", None,  # X X ?
                  "O", "O", None,
                  None, None, None],
        "current_player": "X"
    }

    mcts = MCTS(game, num_iterations=5000, c_param=1.4)
    action = mcts.search(test_state)
    assert action == 2, f"MCTS should pick cell 2 to win immediately, but got {action}!"
    print("Test passed: MCTS found the forced winning move for X.")


if __name__ == "__main__":
    run_tests()
    test_mcts_forced_win()
