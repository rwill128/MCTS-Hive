import copy
import random

class ConnectFourGame:
    ROWS = 6
    COLS = 7

    # We'll label empty as 0, Player1 as 1, Player2 as 2 internally.
    # The interface will still handle 'Player1'/'Player2' strings for you.

    def __init__(self):
        pass

    def getInitialState(self):
        """
        Returns the initial state of the Connect Four game.
        The state includes:
        - A representation of the board as a 2D list (6 rows x 7 cols).
        - The current player to move: 'Player1' or 'Player2'.
        """
        board = [[0 for _ in range(self.COLS)] for _ in range(self.ROWS)]
        return {
            "board": board,
            "current_player": "Player1"
        }

    def getLegalActions(self, state):
        """
        Returns a list of legal actions (columns) for the current player.
        An action is an integer 0..6 representing which column to drop a disc into.
        """
        legal_actions = []
        board = state["board"]
        for col in range(self.COLS):
            # If the top cell is empty, that column is playable
            if board[0][col] == 0:
                legal_actions.append(col)
        return legal_actions

    def applyAction(self, state, action):
        """
        Applies the given action (column index) to the current state and returns the next state.
        The piece will occupy the lowest available row in the chosen column.
        """
        next_state = copy.deepcopy(state)
        board = next_state["board"]
        current_player = next_state["current_player"]

        # Convert 'Player1'/'Player2' to numeric labels 1 or 2
        player_label = 1 if current_player == "Player1" else 2

        # Find the lowest empty row in the chosen column
        for row in range(self.ROWS - 1, -1, -1):
            if board[row][action] == 0:
                board[row][action] = player_label
                break

        # Switch current player
        next_state["current_player"] = (
            "Player2" if current_player == "Player1" else "Player1"
        )

        return next_state

    def isTerminal(self, state):
        """
        Returns True if the game is over (a player has won or the board is full).
        Otherwise, returns False.
        """
        board = state["board"]
        # Check for a four-in-a-row for either player
        if self._checkWinner(board) is not None:
            return True

        # Check for draw (no more moves)
        if all(board[0][c] != 0 for c in range(self.COLS)):
            return True

        return False

    def getReward(self, final_state, root_player):
        """
        Returns the reward for the root_player in the final state:
          +1 if root_player has won,
          -1 if root_player has lost,
           0 for a draw or no winner.
        """
        board = final_state["board"]
        winner = self._checkWinner(board)  # returns 1, 2, or None

        # Map 'Player1'/'Player2' to 1 or 2
        if root_player == "Player1":
            root_label = 1
            opp_label = 2
        else:
            root_label = 2
            opp_label = 1

        if winner == root_label:
            return 1
        elif winner == opp_label:
            return -10
        else:
            # draw or no winner
            return 0

    def getCurrentPlayer(self, state):
        """
        Returns the current player to move ('Player1' or 'Player2').
        """
        return state["current_player"]

    def simulateRandomPlayout(self, state):
        """
        Simulates a random playout from the given state until the game ends.
        Returns the final state.
        """
        temp_state = copy.deepcopy(state)
        while not self.isTerminal(temp_state):
            legal_actions = self.getLegalActions(temp_state)
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            temp_state = self.applyAction(temp_state, action)
        return temp_state

    def printState(self, state):
        """
        Prints the game's board state in a human-friendly format.
        Rows are printed top to bottom.
        """
        board = state["board"]
        print("Current Board State:")
        for row in range(self.ROWS):
            print("|", end="")
            for col in range(self.COLS):
                cell = board[row][col]
                if cell == 0:
                    symbol = " "
                elif cell == 1:
                    symbol = "X"
                else:
                    symbol = "O"
                print(symbol, end="|")
            print()
        print(" 0  1  2  3  4  5  6  ")
        print(f"Current Player: {state['current_player']}\n")

    # ------------------------------
    #        HELPER METHODS
    # ------------------------------
    def _checkWinner(self, board):
        """
        Checks if there's a 4-in-a-row anywhere on the board.
        Returns 1 if Player1 has won, 2 if Player2 has won, or None otherwise.
        """
        # Horizontal check
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                if board[r][c] != 0 and \
                        board[r][c] == board[r][c+1] == board[r][c+2] == board[r][c+3]:
                    return board[r][c]

        # Vertical check
        for c in range(self.COLS):
            for r in range(self.ROWS - 3):
                if board[r][c] != 0 and \
                        board[r][c] == board[r+1][c] == board[r+2][c] == board[r+3][c]:
                    return board[r][c]

        # Diagonal (down-right) check
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                if board[r][c] != 0 and \
                        board[r][c] == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3]:
                    return board[r][c]

        # Diagonal (down-left) check
        for r in range(self.ROWS - 3):
            for c in range(3, self.COLS):
                if board[r][c] != 0 and \
                        board[r][c] == board[r+1][c-1] == board[r+2][c-2] == board[r+3][c-3]:
                    return board[r][c]

        return None
