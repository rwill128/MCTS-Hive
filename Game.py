from ConnectFour.ConnectFour import ConnectFourGame
from HivePocket.DrawGame import drawStatePygame
from HivePocket.HivePocket import HiveGame
from TicTacToe.TicTacToe import TicTacToeGame
from mcts.Mcts import MCTS


def play_with_mcts():
    from time import sleep

    # Create the TicTacToe game
    game = HiveGame()

    # Create two MCTS instances, one for X and one for O
    # (They can have different iteration counts or the same.)
    mcts = MCTS(game, draw_reward=0.1,
                win_reward=1,
                lose_reward=-1,
                num_iterations=200,
                c_param=1.4,
                forced_check_depth=2)

    # Start from the initial state
    state = game.getInitialState()

    # Print the initial empty board
    print("Initial board:")
    game.printState(state)

    while not game.isTerminal(state):

        # Decide which MCTS to use
        best_move = mcts.search(state)

        # Apply the move
        state = game.applyAction(state, best_move)

        print("Took a move in the real game")
        drawStatePygame(state)

        if game.isTerminal(state):
            print("Game over, result: " + game.getGameOutcome(state))


# Run the match
if __name__ == "__main__":
    play_with_mcts()
