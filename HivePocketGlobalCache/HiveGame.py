from HivePocketGlobalCache.GameState import GameState
from HivePocketGlobalCache.HiveRules import HiveRules


class HiveGame:
    INITIAL_PIECES = {
        "Queen": 1,
        "Spider": 2,
        "Beetle": 2,
        "Grasshopper": 3,
        "Ant": 3,
    }
    def __init__(self):
        self.state = GameState()  # Use the new GameState class
        self.rules = HiveRules()   # Create an instance of HiveRules
        self._legal_moves_cache = {} # Caches are here
        self._can_slide_cache = {}
        self._connectivity_cache = {}


    def getInitialState(self):
        return GameState()

    def isTerminal(self, state):
        if self.rules.is_queen_surrounded(state.board, "Player1"):
            return True
        if self.rules.is_queen_surrounded(state.board, "Player2"):
            return True
        if not self.rules.get_legal_actions(state):
            return True
        return False

    def getGameOutcome(self, state):
        p1_surrounded = self.rules.is_queen_surrounded(state.board, "Player1")
        p2_surrounded = self.rules.is_queen_surrounded(state.board, "Player2")
        if p1_surrounded and p2_surrounded:
            print("Ended by draw")
            return "Draw"
        elif p1_surrounded:
            print("Ended because Player1 is surrounded")
            return "Player2"
        elif p2_surrounded:
            print("Ended because Player2 is surrounded")
            return "Player1"
        if not self.rules.get_legal_actions(state):
            current = state.current_player
            opponent = state.get_opponent()
            print("Ended because " + current + " has no legal moves.")
            return opponent
        return None

    def applyAction(self, state, action):
        if not self.rules.is_legal_action(state, action):
            raise ValueError(f"Invalid move: {action} from state: {state}")

        new_state = state.copy()
        new_state.apply_action(action)
        self.clearCaches() #Cache clearing should still happen in HiveGame

        return new_state


    def getLegalActions(self, state):
        # Delegate to HiveRules
        key = self.board_hash(state.board) #Use the same board_hash
        if key in self._legal_moves_cache:
            return self._legal_moves_cache[key]
        legal_moves = self.rules.get_legal_actions(state)
        self._legal_moves_cache[key] = legal_moves
        return legal_moves
    def clearCaches(self):
        self._legal_moves_cache.clear()
        self._can_slide_cache.clear()
        self._connectivity_cache.clear()


    def evaluateState(self, state):
        #This can also be refactored later on to make better use of HiveRules.
        outcome = self.getGameOutcome(state)
        if outcome is not None:
            if outcome == "Player1": return 10000