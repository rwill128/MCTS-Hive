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

RATING_DROP_THRESHOLD = 1180     # Agents below this rating are removed
CHECKPOINT_FREQUENCY = 5         # How often (in rounds) to log current leader
WEIGHT_SCALE = 20             # Scaling factor for selecting a challenger
# (higher rating => higher probability to be challenger)

def initialize_elo(agent_count, initial_rating=1200):
    """
    Return a dict: agent_id -> current Elo rating
    """
    return {agent_id: initial_rating for agent_id in range(agent_count)}

def update_elo(ratingA, ratingB, scoreA, scoreB, K=16):
    """
    Standard Elo update for a single match.
      - ratingA, ratingB: current Elo ratings
      - scoreA, scoreB in [0..1] and scoreA + scoreB = 1.0
        (e.g. 1.0 vs 0.0, 0.5 vs 0.5, 0.75 vs 0.25, etc.)
    Returns updatedRatingA, updatedRatingB
    """
    expectedA = 1.0 / (1.0 + 10 ** ((ratingB - ratingA) / 400))
    expectedB = 1.0 - expectedA

    newA = ratingA + K * (scoreA - expectedA)
    newB = ratingB + K * (scoreB - expectedB)
    return newA, newB

def play_single_game(game, agentA, agentB, agentA_is_player1=True):
    """
    Plays exactly one game of Connect Four:
      - If agentA_is_player1=True, agentA is "Player1" and agentB is "Player2".
      - Otherwise agentA is "Player2" and agentB is "Player1".
    Returns "Player1", "Player2", or "Draw"
    """
    state = game.getInitialState()

    # We can store references for convenience:
    player1_agent = agentA if agentA_is_player1 else agentB
    player2_agent = agentB if agentA_is_player1 else agentA

    while not game.isTerminal(state):
        current_player = game.getCurrentPlayer(state)
        if current_player == "Player1":
            action = player1_agent.search(state)
        else:
            action = player2_agent.search(state)

        state = game.applyAction(state, action)
        game.printState(state)

    outcome = game.getGameOutcome(state)  # "Player1", "Player2", or "Draw"
    return outcome

def play_two_game_match(game, champion, challenger):
    """
    Plays two games between champion and challenger:
      1) champion as Player1, challenger as Player2
      2) champion as Player2, challenger as Player1

    Returns a tuple (champion_points, challenger_points) in [0..2].

    Point system per game:
      - Win = 1 point
      - Draw = 0.5 points
      - Loss = 0 points
    """
    # Game 1: champion -> Player1
    outcome_g1 = play_single_game(game, champion, challenger, agentA_is_player1=True)
    # Game 2: champion -> Player2
    outcome_g2 = play_single_game(game, champion, challenger, agentA_is_player1=False)

    # Let's compute champion's total from these 2 games
    champion_points = 0.0
    challenger_points = 0.0

    # Evaluate outcome of game1
    if outcome_g1 == "Player1":
        # champion won
        champion_points += 1.0
    elif outcome_g1 == "Player2":
        # challenger won
        challenger_points += 1.0
    else:  # "Draw"
        champion_points += 0.5
        challenger_points += 0.5

    # Evaluate outcome of game2
    if outcome_g2 == "Player2":
        # champion won (because champion was Player2)
        champion_points += 1.0
    elif outcome_g2 == "Player1":
        # challenger won (champion was Player2 => if Player1 wins, challenger wins)
        challenger_points += 1.0
    else:  # "Draw"
        champion_points += 0.5
        challenger_points += 0.5

    return champion_points, challenger_points

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
                        num_iterations=1000,
                        do_forced_move_check=True,
                        forced_check_depth=6# Adjust as desired
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
        f"Starting a King-of-the-Hill style tournament with two-game matches.\n"
        f"Drop-threshold ELO={RATING_DROP_THRESHOLD}.\n"
        "Each match is 2 games (Champion as Player1, then Player2).\n"
        "Elo is updated once per match, based on the total match score.\n"
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


            challengers = list(active_agents)
            weights = [math.exp(elo_ratings[ch] / WEIGHT_SCALE) for ch in challengers]

            # 1. Identify champion: the top-rated agent among active_agents
            champion_id = random.choices(challengers, weights=weights, k=1)[0]
            champion_desc = describe_agent(champion_id, param_combos, elo_ratings)

            # 2. Build list of possible challengers (active, not champion)


            # Weighted random selection:

            challenger_id = random.choices(challengers, weights=weights, k=1)[0]
            challenger_desc = describe_agent(challenger_id, param_combos, elo_ratings)

            logging.info(
                f"Round {round_counter} MATCH:\n"
                f"   CHAMPION: {champion_desc}\n"
                f"   CHALLENGER (weighted-random): {challenger_desc}"
            )

            # 3. Play a two-game match
            champion_points, challenger_points = play_two_game_match(
                game, agents[champion_id], agents[challenger_id]
            )

            # champion_points + challenger_points = 2.0 (2 games)
            # Convert to fraction for Elo update
            champ_fraction = champion_points / 2.0   # in [0..1]
            chall_fraction = challenger_points / 2.0 # in [0..1]

            # We'll log the specific 2-game result
            logging.info(
                f"Match result => Champion scored {champion_points:.1f}, "
                f"Challenger scored {challenger_points:.1f} (out of 2 games)."
            )

            # 4. Elo update
            oldChamp, oldChall = elo_ratings[champion_id], elo_ratings[challenger_id]
            newChamp, newChall = update_elo(oldChamp, oldChall, champ_fraction, chall_fraction)
            elo_ratings[champion_id] = newChamp
            elo_ratings[challenger_id] = newChall

            logging.info(
                f"   {champion_desc} was {oldChamp:.1f} -> {newChamp:.1f}\n"
                f"   {challenger_desc} was {oldChall:.1f} -> {newChall:.1f}"
            )

            # 5. Check if either agent's rating fell below the threshold
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

            # 6. Periodically log the overall top agent
            if round_counter % CHECKPOINT_FREQUENCY == 0:
                sorted_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
                overall_leader = sorted_elo[0][0]
                logging.info(
                    f"--- After {round_counter} matches, overall top rating: "
                    f"{describe_agent(overall_leader, param_combos, elo_ratings)} ---\n"
                )

            # Optional delay
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
