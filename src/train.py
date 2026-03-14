"""
Training script for the Tic-Tac-Toe RL agent.

The agent learns through self-play: it controls both X and O,
updating Q-values from both perspectives. Over time, it figures
out the optimal strategy on its own.

Usage:
    python src/train.py --episodes 50000 --lr 0.1 --epsilon 0.3
"""

import argparse
import sys
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so it works without display
import matplotlib.pyplot as plt

# add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import TicTacToeEnv
from src.agent import QLearningAgent, RandomAgent


def train_self_play(agent, episodes=50000, eval_every=1000, eval_games=500):
    """
    Train agent through self-play.

    The same Q-learning agent plays as both X and O. This way it
    learns to handle both attacking and defending.

    Returns:
        history: dict with win rates and other metrics over training
    """
    env = TicTacToeEnv()
    history = {"episodes": [], "win_rate_vs_random": [], "draw_rate": []}

    for episode in range(1, episodes + 1):
        state = env.reset()

        # store the trajectory for both players so we can update after the game
        trajectory = {1: [], -1: []}

        while not env.done:
            valid_actions = env.get_valid_actions()
            player = env.current_player

            action = agent.choose_action(state, valid_actions, training=True)
            next_state, reward, done, info = env.step(action)

            # save this transition
            trajectory[player].append({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            })

            state = next_state

        # now update Q-values for both players
        # the tricky part: when player X wins, player O should get a negative reward
        winner = info["winner"]

        for player in [1, -1]:
            for i, step in enumerate(trajectory[player]):
                if step["done"]:
                    # this player made the last move
                    if winner == player:
                        final_reward = 1.0   # they won
                    elif winner == 0:
                        final_reward = 0.5   # draw
                    else:
                        final_reward = -1.0  # they lost
                    agent.update(
                        step["state"], step["action"], final_reward,
                        step["next_state"], [], done=True
                    )
                elif i == len(trajectory[player]) - 1 and not step["done"]:
                    # last move by this player, but game ended on opponent's turn
                    if winner == player:
                        final_reward = 1.0
                    elif winner == 0:
                        final_reward = 0.5
                    else:
                        final_reward = -1.0
                    agent.update(
                        step["state"], step["action"], final_reward,
                        step["next_state"], [], done=True
                    )
                else:
                    # mid-game move
                    next_step = trajectory[player][i + 1]
                    next_valid = env.board  # approximate — we just need non-terminal
                    agent.update(
                        step["state"], step["action"], 0.0,
                        next_step["state"],
                        [j for j in range(9) if next_step["state"][j] == 0],
                        done=False
                    )

        agent.decay_epsilon()

        # periodic evaluation against random opponent
        if episode % eval_every == 0:
            win_rate, draw_rate = evaluate(agent, num_games=eval_games)
            history["episodes"].append(episode)
            history["win_rate_vs_random"].append(win_rate)
            history["draw_rate"].append(draw_rate)
            print(
                f"Episode {episode:>6}/{episodes} | "
                f"Win Rate: {win_rate:.1%} | "
                f"Draw Rate: {draw_rate:.1%} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Q-table size: {len(agent.q_table)}"
            )

    return history


def evaluate(agent, num_games=500):
    """Test the agent against a random opponent."""
    env = TicTacToeEnv()
    random_opponent = RandomAgent()
    wins = 0
    draws = 0

    for game in range(num_games):
        state = env.reset()

        # alternate who goes first
        agent_player = 1 if game % 2 == 0 else -1

        while not env.done:
            valid_actions = env.get_valid_actions()

            if env.current_player == agent_player:
                action = agent.choose_action(state, valid_actions, training=False)
            else:
                action = random_opponent.choose_action(state, valid_actions)

            state, _, done, info = env.step(action)

        if info["winner"] == agent_player:
            wins += 1
        elif info["winner"] == 0:
            draws += 1

    return wins / num_games, draws / num_games


def plot_training(history, save_path="images/training_curve.png"):
    """Plot the win rate over training episodes."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(history["episodes"], history["win_rate_vs_random"],
            label="Win Rate vs Random", color="#2ecc71", linewidth=2)
    ax.plot(history["episodes"], history["draw_rate"],
            label="Draw Rate", color="#3498db", linewidth=2, linestyle="--")

    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Rate")
    ax.set_title("Agent Performance Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curve saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a Tic-Tac-Toe RL agent")
    parser.add_argument("--episodes", type=int, default=50000, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (alpha)")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.3, help="Initial exploration rate")
    args = parser.parse_args()

    print("=" * 50)
    print("  Tic-Tac-Toe Q-Learning Agent")
    print("=" * 50)
    print(f"  Episodes:  {args.episodes}")
    print(f"  LR:        {args.lr}")
    print(f"  Gamma:     {args.gamma}")
    print(f"  Epsilon:   {args.epsilon}")
    print("=" * 50)
    print()

    agent = QLearningAgent(
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon=args.epsilon
    )

    history = train_self_play(agent, episodes=args.episodes)

    # save everything
    agent.save("models/q_table.pkl")
    plot_training(history)

    print("\nDone! Run 'python src/play.py' to play against the agent.")


if __name__ == "__main__":
    main()
