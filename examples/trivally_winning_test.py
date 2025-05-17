from HivePocket.HivePocket import HiveGame


def make_winning_for_player1(game):
    """
    Return a state where Player2's queen is fully surrounded, so the outcome is 'Player1'.
    current_player = 'Player2', so 'Player2' is about to move in a lost position.
    """
    state = {
        "board": {},
        "current_player": "Player2",  # Next to move
        "pieces_in_hand": {
            "Player1": game.INITIAL_PIECES.copy(),
            "Player2": game.INITIAL_PIECES.copy()
        },
        "move_number": 6
    }

    # Player2's queen at (0,0)
    state["board"][(0, 0)] = [("Player2", "Queen")]

    # Surround the queen with 6 pieces from Player1
    for (dq, dr) in game.DIRECTIONS:
        surround_cell = (dq, dr)
        state["board"][surround_cell] = [("Player1", "Ant")]

    return state

if __name__ == "__main__":
    game = HiveGame()
    winning_state = make_winning_for_player1(game)
    score = game.evaluateState(winning_state)
    print("Winning-for-Player1 scenario => evaluateState =", score)
