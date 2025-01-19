from ConnectFour.ConnectFour import ConnectFourGame
from HivePocket.HivePocket import HiveGame
from TicTacToe.TicTacToe import TicTacToeGame
from mcts.Mcts import MCTS


def play_tictactoe_mcts():
    from time import sleep

    # Create the TicTacToe game
    game = ConnectFourGame()

    # Create two MCTS instances, one for X and one for O
    # (They can have different iteration counts or the same.)
    mcts = MCTS(game, draw_reward=0.1,
                win_reward=2,
                lose_reward=-1,
                num_iterations=2000,
                c_param=1.0,
                forced_check_depth=9)

    # Start from the initial state
    state = game.getInitialState()

    # Print the initial empty board
    print("Initial board:")
    game.printState(state)

    while not game.isTerminal(state):
        current_player = game.getCurrentPlayer(state)

        # Decide which MCTS to use
        best_move = mcts.search(state)

        # Apply the move
        state = game.applyAction(state, best_move)

        # Print the updated board
        print(f"Player {current_player} moves at cell {best_move}:")
        game.printState(state)
        # sleep(1) # optional: to slow down the output

    print("Result is:" + game.getGameOutcome(state))

# Run the match
if __name__ == "__main__":
    play_tictactoe_mcts()
