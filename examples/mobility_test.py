from HivePocket.HivePocket import HiveGame


def make_early_game_mobility_test(game):
    """
    Minimal board where Player1 has 3 pieces out and Player2 has 1, so that the
    'early_factor' part might favor Player1 if current_player=Player1 and the
    code interprets 'having more pieces out early' as beneficial.
    """
    state = {
        "board": {},
        "current_player": "Player1",
        "pieces_in_hand": {
            "Player1": game.INITIAL_PIECES.copy(),
            "Player2": game.INITIAL_PIECES.copy()
        },
        "move_number": 4  # early-ish
    }

    # Put down P1’s Queen, Ant, Beetle around (0,0) so it's connected
    state["board"][(0, 0)] = [("Player1", "Queen")]
    state["board"][(1, 0)] = [("Player1", "Ant")]
    state["board"][(1, -1)] = [("Player1", "Beetle")]

    # Put down P2’s Queen only
    state["board"][(-1, 0)] = [("Player2", "Queen")]

    # Adjust 'pieces_in_hand' accordingly to reflect that we've used these pieces
    state["pieces_in_hand"]["Player1"]["Queen"] -= 1
    state["pieces_in_hand"]["Player1"]["Ant"]   -= 1
    state["pieces_in_hand"]["Player1"]["Beetle"]-= 1

    state["pieces_in_hand"]["Player2"]["Queen"] -= 1

    return state

if __name__ == "__main__":
    game = HiveGame()
    early_test_state = make_early_game_mobility_test(game)

    # Emphasize early_factor so we can see it matter more
    custom_weights = {
        "queen_factor": 10,
        "liberties_factor": 2,
        "mobility_factor": 1,
        "early_factor": 20,
    }

    score = game.evaluateState(early_test_state, weights=custom_weights)
    print("Early game mobility test => evaluateState =", score)
