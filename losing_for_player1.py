from HivePocket.HivePocket import HiveGame


def make_losing_for_player1(game):
    """
    Return a state where Player1's queen is fully surrounded by Player2's pieces.
    By the code's current convention, evaluateState will return +10000 if
    'Player2' is the outcome and 'Player1' is current_player.
    (Often you'd expect a negative value in that scenario, so be mindful if you
     want the usual perspective-based scoring.)
    """
    state = {
        "board": {},
        "current_player": "Player1",  # Next to move is Player1
        "pieces_in_hand": {
            "Player1": game.INITIAL_PIECES.copy(),
            "Player2": game.INITIAL_PIECES.copy()
        },
        "move_number": 6  # arbitrary
    }

    # Player1's queen at (0,0)
    state["board"][(0, 0)] = [("Player1", "Queen")]

    # Surround the queen with 6 pieces from Player2
    for (dq, dr) in game.DIRECTIONS:  # e.g. (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)
        surround_cell = (0 + dq, 0 + dr)
        state["board"][surround_cell] = [("Player2", "Ant")]  # arbitrary insect

    return state

if __name__ == "__main__":
    game = HiveGame()
    losing_state = make_losing_for_player1(game)
    score = game.evaluateState(losing_state)
    print("Losing-for-Player1 scenario => evaluateState =", score)
