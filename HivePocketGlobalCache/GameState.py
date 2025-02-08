from HivePocket.HivePocket import HiveGame


class GameState:
    def __init__(self, board=None, current_player="Player1", pieces_in_hand=None, move_number=0):
        self.board = board if board is not None else {}
        self.current_player = current_player
        self.pieces_in_hand = pieces_in_hand if pieces_in_hand is not None else {
            "Player1": HiveGame.INITIAL_PIECES.copy(),  # Use class-level constant
            "Player2": HiveGame.INITIAL_PIECES.copy(),
        }
        self.move_number = move_number

    def copy(self):
        return GameState(
            board={coord: stack[:] for coord, stack in self.board.items()},
            current_player=self.current_player,
            pieces_in_hand={
                player: pieces.copy() for player, pieces in self.pieces_in_hand.items()
            },
            move_number=self.move_number,
        )

    def apply_action(self, action):
        # NO validation here - assumes the action is legal
        if action[0] == "PLACE":
            _, insectType, (q, r) = action
            self.pieces_in_hand[self.current_player][insectType] -= 1
            if (q, r) not in self.board:
                self.board[(q, r)] = []
            self.board[(q, r)].append((self.current_player, insectType))
        elif action[0] == "MOVE":
            _, (fq, fr), (tq, tr) = action
            piece = self.board[(fq, fr)].pop()
            if not self.board[(fq, fr)]:
                del self.board[(fq, fr)]
            self.board.setdefault((tq, tr), []).append(piece)

        self.current_player = "Player2" if self.current_player == "Player1" else "Player1"
        self.move_number += 1

    def get_current_player(self):
        return self.current_player

    def get_opponent(self):
        return "Player2" if self.current_player == "Player1" else "Player1"