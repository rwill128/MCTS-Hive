import random

class TicTacToeGame:
    def __init__(self):
        pass

    def getInitialState(self):
        """
        Returns the initial state of the TicTacToe board.
        Board is 3x3, stored as a list of length 9: index 0..8
          0 | 1 | 2
          3 | 4 | 5
          6 | 7 | 8
        current_player is 'X' or 'O'.
        """
        return {
            "board": [None] * 9,  # 9 empty cells
            "current_player": 'X'
        }

    def getLegalActions(self, state):
        """
        Returns all empty board positions where a move can be placed.
        Each action is an integer index 0..8.
        """
        board = state["board"]
        return [i for i, cell in enumerate(board) if cell is None]

    def applyAction(self, state, action):
        """
        Applies the given action (board index) for the current player
        and returns the resulting next state.
        """
        board = list(state["board"])  # make a shallow copy
        current_player = state["current_player"]

        # Place current player's mark on the chosen cell
        board[action] = current_player

        # Switch player
        next_player = 'O' if current_player == 'X' else 'X'

        return {
            "board": board,
            "current_player": next_player
        }

    def isTerminal(self, state):
        """
        Returns True if the game is over (win or draw), False otherwise.
        """
        return self._checkWinner(state) is not None or \
            all(cell is not None for cell in state["board"])

    def getReward(self, final_state, root_player):
        winner = self._checkWinner(final_state)
        if winner is None:
            return 0
        elif winner == root_player:
            return 1
        else:
            return -1

    def getCurrentPlayer(self, state):
        """
        Returns the current player to move ('X' or 'O').
        """
        return state["current_player"]

    def simulateRandomPlayout(self, state):
        """
        From the given state, play random legal moves until the game ends.

        Modification: If the current player has a winning move, take it
        immediately instead of choosing randomly.
        """
        temp_state = {
            "board": list(state["board"]),
            "current_player": state["current_player"]
        }

        while not self.isTerminal(temp_state):
            legal_actions = self.getLegalActions(temp_state)
            if not legal_actions:
                break  # no moves => terminal

            current_player = temp_state["current_player"]

            # 1) Attempt to find a winning move for the current player
            winning_move = None
            for action in legal_actions:
                # Test applying the action
                new_state = self.applyAction(temp_state, action)
                if self.isTerminal(new_state):
                    winning_move = action
                    break

            # 2) If a winning move is found, use it;
            #    Otherwise pick a random move
            if winning_move is not None:
                action = winning_move
            else:
                action = random.choice(legal_actions)

            temp_state = self.applyAction(temp_state, action)

        return temp_state

    def printState(self, state):
        """
        Print the TicTacToe board in a human-friendly format.
        E.g.:

         X | O |
        ---+---+---
           | X |
        ---+---+---
           |   | O
        """
        board = state["board"]
        symbols = [cell if cell is not None else ' ' for cell in board]
        for row in range(3):
            row_cells = symbols[3*row : 3*row+3]
            print(" {} | {} | {} ".format(*row_cells))
            if row < 2:
                print("---+---+---")
        print()

    # ---------------------------------------------------------
    # Internal helper to check if there's a winner
    # Returns 'X', 'O', or None if no winner yet.
    # ---------------------------------------------------------
    def _checkWinner(self, state):
        board = state["board"]
        # Possible winning lines (indices)
        wins = [
            (0,1,2), (3,4,5), (6,7,8),  # rows
            (0,3,6), (1,4,7), (2,5,8),  # cols
            (0,4,8), (2,4,6)            # diagonals
        ]
        for (a,b,c) in wins:
            if board[a] is not None and board[a] == board[b] == board[c]:
                return board[a]  # 'X' or 'O'
        return None
