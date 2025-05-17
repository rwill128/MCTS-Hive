from HivePocket.HivePocket import HiveGame


def make_almost_surrounded_p2_queen(game):
    """
    Player2’s queen is placed with 5 neighbors from Player1.
    If current_player is Player1, we should see a strongly positive heuristic
    (under typical 'winning = plus' logic).
    """
    state = {
        "board": {},
        "current_player": "Player1",
        "pieces_in_hand": {
            "Player1": game.INITIAL_PIECES.copy(),
            "Player2": game.INITIAL_PIECES.copy()
        },
        "move_number": 5
    }

    # Place Player2's queen at (0,0)
    state["board"][(0, 0)] = [("Player2", "Queen")]

    # Surround 5 neighbors with Player1 pieces
    directions = list(game.DIRECTIONS)
    for i in range(5):
        dq, dr = directions[i]
        cell = (dq, dr)
        state["board"][cell] = [("Player1", "Ant")]

    return state

if __name__ == "__main__":
    game = HiveGame()
    good_state = make_almost_surrounded_p2_queen(game)
    custom_weights = {
        "queen_factor": 50,
        "liberties_factor": 10,
        "mobility_factor": 3,
        "early_factor": 2,
    }
    score = game.evaluateState(good_state, weights=custom_weights)
    print("Almost-surrounded P2 queen => evaluateState =", score)
