from TicTacToe.TicTacToe import TicTacToeGame
from mcts.Mcts import MCTS


def play_tictactoe_mcts():
    from time import sleep

    # Create the TicTacToe game
    game = TicTacToeGame()

    # Create two MCTS instances, one for X and one for O
    # (They can have different iteration counts or the same.)
    mcts = MCTS(game, num_iterations=20000, c_param=1.4)

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

    # Game is over - determine outcome
    winner = game._checkWinner(state)
    if winner is None:
        print("It's a draw!")
    else:
        print(f"Player {winner} wins!")

# Run the match
if __name__ == "__main__":
    play_tictactoe_mcts()
