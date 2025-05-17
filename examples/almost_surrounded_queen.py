from HivePocket.HivePocket import HiveGame


def make_almost_surrounded_p1_queen(game):
    """
    Player1's queen is placed and 5 neighbors belong to Player2.
    The 6th neighbor is free, so not fully surrounded => game not over yet.
    This should produce a large negative 'queen factor' if the code
    is from the perspective of Player1 (assuming you'd normally want negative if it's bad).
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

    # Place Player1's queen at (0,0)
    state["board"][(0, 0)] = [("Player1", "Queen")]

    # Surround 5 of the adjacent cells
    directions = list(game.DIRECTIONS)
    for i in range(5):
        dq, dr = directions[i]
        cell = (dq, dr)
        state["board"][cell] = [("Player2", "Ant")]

    # The 6th cell remains empty
    # state["board"] is missing the 6th neighbor, so that is effectively free.

    return state

if __name__ == "__main__":
    game = HiveGame()
    almost_surrounded_state = make_almost_surrounded_p1_queen(game)

    # Suppose we want to highlight the queen/libraries factor in the heuristic:
    custom_weights = {
        "queen_factor": 50,
        "liberties_factor": 10,
        "mobility_factor": 3,
        "early_factor": 2,
    }

    score = game.evaluateState(almost_surrounded_state, weights=custom_weights)
    print("Almost-surrounded P1 queen => evaluateState =", score)
