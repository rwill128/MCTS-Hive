import math
import random
import copy
import time
import logging

# If you haven't already configured logging elsewhere:
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

from ConnectFour.ConnectFour import ConnectFourGame
from mcts.Mcts import MCTS

def initialize_elo(agent_count, initial_rating=1200):
    """
    Return a dict: agent_id -> current Elo rating
    """
    return {agent_id: initial_rating for agent_id in range(agent_count)}

def update_elo(ratingA, ratingB, scoreA, scoreB, K=16):
    """
    Standard Elo update.
      - ratingA, ratingB: current Elo ratings
      - scoreA, scoreB in {0, 0.5, 1.0}
    Returns updatedRatingA, updatedRatingB
    """
    expectedA = 1.0 / (1.0 + 10 ** ((ratingB - ratingA) / 400))
    expectedB = 1.0 - expectedA

    newA = ratingA + K * (scoreA - expectedA)
    newB = ratingB + K * (scoreB - expectedB)
    return newA, newB

def play_single_match(game, agent1, agent2):
    """
    Plays exactly one match of Connect Four:
      - agent1 controls "Player1"
      - agent2 controls "Player2"
    Returns "Player1", "Player2", or "Draw"
    """
    state = game.getInitialState()

    while not game.isTerminal(state):
        current_player = game.getCurrentPlayer(state)
        if current_player == "Player1":
            action = agent1.search(state)
        else:
            action = agent2.search(state)

        state = game.applyAction(state, action)

    outcome = game.getGameOutcome(state)  # "Player1", "Player2", or "Draw"
    return outcome

def describe_agent(agent_id, param_combos, elo_ratings):
    """
    Returns a string describing one agent: its ID, parameters, and Elo rating.
    """
    c, w, l, d, _ = param_combos[agent_id]
    rating = elo_ratings[agent_id]
    return f"Agent#{agent_id} [c={c}, w={w}, l={l}, d={d}, ELO={rating:.1f}]"

def main():
    # Create the base game object
    game = ConnectFourGame()

    # 1) Generate a variety of parameter combos for MCTS
    param_combos = []
    for c in [0.5, 1.0, 1.4, 2.0, 2.5, 3.0]:
        for w in [1, 2, 5]:
            for l in [-1, -3, -10, -100]:
                for d in [0, 0.1, 0.5]:
                    # Build an MCTS agent
                    agent = MCTS(
                        game=game,
                        win_reward=w,
                        lose_reward=l,
                        draw_reward=d,
                        c_param=c,
                        num_iterations=100000  # Adjust as desired
                    )
                    param_combos.append((c, w, l, d, agent))

    # 2) Extract just the agent objects in a list
    agents = [combo[4] for combo in param_combos]
    agent_count = len(agents)
    logging.info(f"Created {agent_count} agents with distinct parameter combos.")

    # 3) Initialize Elo ratings
    elo_ratings = initialize_elo(agent_count, initial_rating=1200)

    # Log the initial list of agents and their parameters
    logging.info("Initial agent configurations (ELO=1200 for all):")
    for agent_id in range(agent_count):
        logging.info(describe_agent(agent_id, param_combos, elo_ratings))

    logging.info("Starting an indefinite tournament... press Ctrl+C to stop at any time.")

    round_counter = 0
    try:
        while True:
            round_counter += 1

            # Randomly pick two different agents
            idx1, idx2 = random.sample(range(agent_count), 2)

            # Log info about who is playing this round
            logging.info(
                f"Round {round_counter} match-up:\n"
                f"   {describe_agent(idx1, param_combos, elo_ratings)}\n"
                f" vs\n"
                f"   {describe_agent(idx2, param_combos, elo_ratings)}"
            )

            outcome = play_single_match(game, agents[idx1], agents[idx2])

            if outcome == "Player1":
                scoreA, scoreB = 1.0, 0.0
            elif outcome == "Player2":
                scoreA, scoreB = 0.0, 1.0
            else:  # Draw
                scoreA, scoreB = 0.5, 0.5

            # Update Elo
            oldA, oldB = elo_ratings[idx1], elo_ratings[idx2]
            newA, newB = update_elo(oldA, oldB, scoreA, scoreB)
            elo_ratings[idx1] = newA
            elo_ratings[idx2] = newB

            # Log the result & rating updates
            logging.info(
                f"Result: {outcome}.\n"
                f"   {describe_agent(idx1, param_combos, elo_ratings)} was {oldA:.1f} -> {newA:.1f}\n"
                f"   {describe_agent(idx2, param_combos, elo_ratings)} was {oldB:.1f} -> {newB:.1f}"
            )

            # Every 5 rounds, log the current leader or top few
            if round_counter % 5 == 0:
                sorted_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
                leader_id = sorted_elo[0][0]
                logging.info(f"--- After {round_counter} rounds, current leader: {describe_agent(leader_id, param_combos, elo_ratings)} ---\n")

            # Optional delay to avoid spamming too fast
            # time.sleep(0.5)

    except KeyboardInterrupt:
        logging.info("\nTournament stopped by user.\n")

    # Final Elo standings
    sorted_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    logging.info("Final ELO standings (highest first):")
    for agent_id, rating in sorted_elo:
        logging.info(describe_agent(agent_id, param_combos, elo_ratings))

if __name__ == "__main__":
    main()
