import random
import pickle
import os
from collections import defaultdict


class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.3):
        """
        Args:
            learning_rate: how aggressively to update Q-values (alpha)
            discount_factor: how much to weight future rewards (gamma)
            epsilon: probability of taking a random action (exploration)
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q-table: maps (state, action) -> expected reward
        # defaultdict so we don't have to initialize every state
        self.q_table = defaultdict(float)

    def choose_action(self, state, valid_actions, training=True):
        """
        Pick an action using epsilon-greedy strategy.

        During training: explore randomly with probability epsilon.
        During play: always pick the best known action.
        """
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # pick the action with the highest Q-value
        q_values = [self.q_table[(state, a)] for a in valid_actions]
        max_q = max(q_values)

        # if there's a tie, pick randomly among the best
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_valid_actions, done):
        """
        Standard Q-learning update.

        Q(s,a) = Q(s,a) + lr * [reward + gamma * max Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[(state, action)]

        if done:
            target = reward
        else:
            # look ahead: what's the best we can do from the next state?
            future_q = max(
                [self.q_table[(next_state, a)] for a in next_valid_actions],
                default=0.0
            )
            target = reward + self.gamma * future_q

        # update towards the target
        self.q_table[(state, action)] += self.lr * (target - current_q)

    def decay_epsilon(self, decay_rate=0.9999, min_epsilon=0.01):
        """Gradually reduce exploration over time."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def save(self, filepath="models/q_table.pkl"):
        """Save the Q-table to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table saved to {filepath} ({len(self.q_table)} entries)")

    def load(self, filepath="models/q_table.pkl"):
        """Load a previously saved Q-table."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(float, data)
        print(f"Q-table loaded from {filepath} ({len(self.q_table)} entries)")


class RandomAgent:
    """A baseline agent that picks random moves. Used for evaluation."""

    def choose_action(self, state, valid_actions, training=False):
        return random.choice(valid_actions)
