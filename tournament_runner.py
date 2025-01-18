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

RATING_DROP_THRESHOLD = 1000  # Agents below this rating are removed
CHECKPOINT_FREQUENCY = 5      # How often (in rounds) to log current leader

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

    # Create a set of active agents. We remove agents from here if they drop below threshold.
    active_agents = set(range(agent_count))

    # Log the initial list of agents and their parameters
    logging.info("Initial agent configurations (ELO=1200 for all):")
    for agent_id in range(agent_count):
        logging.info(describe_agent(agent_id, param_combos, elo_ratings))

    logging.info(
        f"Starting a King-of-the-Hill style tournament with drop-threshold ELO={RATING_DROP_THRESHOLD}.\n"
        "In each round, the current top-rated agent faces a random challenger.\n"
        "Agents below this rating are removed from further competition.\n"
        "Press Ctrl+C to stop at any time."
    )

    round_counter = 0
    try:
        while True:
            round_counter += 1

            # If fewer than 2 agents remain, break out
            if len(active_agents) < 2:
                logging.info("Not enough active agents remaining to continue.")
                break

            # Identify champion: the top-rated agent among active_agents
            champion_id = max(active_agents, key=lambda aid: elo_ratings[aid])
            champion_desc = describe_agent(champion_id, param_combos, elo_ratings)

            # Pick one random challenger that is not the champion
            if len(active_agents) == 1:
                # Only champion left
                logging.info("No challenger available; stopping.")
                break
            possible_challengers = list(active_agents - {champion_id})
            challenger_id = random.choice(possible_challengers)
            challenger_desc = describe_agent(challenger_id, param_combos, elo_ratings)

            logging.info(
                f"Round {round_counter} match-up:\n"
                f"   CHAMPION: {champion_desc}\n"
                f" vs\n"
                f"   CHALLENGER: {challenger_desc}"
            )

            # The champion plays as Player1, challenger as Player2
            outcome = play_single_match(game, agents[champion_id], agents[challenger_id])

            if outcome == "Player1":
                # champion (idxChampion) wins
                scoreChamp, scoreChallenger = 1.0, 0.0
            elif outcome == "Player2":
                # challenger (idxChallenger) wins
                scoreChamp, scoreChallenger = 0.0, 1.0
            else:  # Draw
                scoreChamp, scoreChallenger = 0.5, 0.5

            # Update Elo
            oldChamp, oldChall = elo_ratings[champion_id], elo_ratings[challenger_id]
            newChamp, newChall = update_elo(oldChamp, oldChall, scoreChamp, scoreChallenger)
            elo_ratings[champion_id] = newChamp
            elo_ratings[challenger_id] = newChall

            # Log the result & rating updates
            logging.info(
                f"Result: {outcome}.\n"
                f"   {champion_desc} was {oldChamp:.1f} -> {newChamp:.1f}\n"
                f"   {challenger_desc} was {oldChall:.1f} -> {newChall:.1f}"
            )

            # Check if either agent's rating fell below the threshold
            # If so, remove them from active_agents
            if elo_ratings[champion_id] < RATING_DROP_THRESHOLD and champion_id in active_agents:
                active_agents.remove(champion_id)
                logging.info(
                    f"*** {describe_agent(champion_id, param_combos, elo_ratings)} "
                    f"has fallen below {RATING_DROP_THRESHOLD} and is removed from competition."
                )
            if elo_ratings[challenger_id] < RATING_DROP_THRESHOLD and challenger_id in active_agents:
                active_agents.remove(challenger_id)
                logging.info(
                    f"*** {describe_agent(challenger_id, param_combos, elo_ratings)} "
                    f"has fallen below {RATING_DROP_THRESHOLD} and is removed from competition."
                )

            # Every X rounds, log the current leader or top few
            if round_counter % CHECKPOINT_FREQUENCY == 0:
                sorted_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
                leader_id = sorted_elo[0][0]
                logging.info(
                    f"--- After {round_counter} rounds, overall top rating: "
                    f"{describe_agent(leader_id, param_combos, elo_ratings)} ---\n"
                )

            # Optional delay to slow things down
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
