"""
Basic tests for the Tic-Tac-Toe environment.
Run with: pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import TicTacToeEnv
from src.agent import QLearningAgent, RandomAgent


def test_initial_board_is_empty():
    env = TicTacToeEnv()
    state = env.reset()
    assert all(cell == 0 for cell in state)


def test_valid_actions_decreases():
    env = TicTacToeEnv()
    env.reset()
    assert len(env.get_valid_actions()) == 9

    env.step(0)
    assert len(env.get_valid_actions()) == 8

    env.step(4)
    assert len(env.get_valid_actions()) == 7


def test_player_alternates():
    env = TicTacToeEnv()
    env.reset()
    assert env.current_player == 1

    env.step(0)
    assert env.current_player == -1

    env.step(1)
    assert env.current_player == 1


def test_win_detection():
    """X plays top row and should win."""
    env = TicTacToeEnv()
    env.reset()

    env.step(0)  # X at 0
    env.step(3)  # O at 3
    env.step(1)  # X at 1
    env.step(4)  # O at 4
    _, reward, done, info = env.step(2)  # X at 2 — top row complete

    assert done is True
    assert info["winner"] == 1
    assert reward == 1.0


def test_draw_detection():
    """Fill the board with no winner."""
    env = TicTacToeEnv()
    env.reset()

    # this sequence produces a draw:
    # X O X
    # X X O
    # O X O
    moves = [0, 1, 2, 4, 3, 5, 7, 6, 8]
    for i, move in enumerate(moves):
        _, reward, done, info = env.step(move)

    assert done is True
    assert info["winner"] == 0


def test_invalid_move_raises():
    env = TicTacToeEnv()
    env.reset()
    env.step(4)

    try:
        env.step(4)  # same position
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_random_agent_always_picks_valid():
    env = TicTacToeEnv()
    agent = RandomAgent()

    for _ in range(100):
        env.reset()
        while not env.done:
            valid = env.get_valid_actions()
            action = agent.choose_action(env.get_state(), valid)
            assert action in valid
            env.step(action)


def test_q_agent_learns_something():
    """After some training, agent should beat random more than 50% of the time."""
    agent = QLearningAgent(learning_rate=0.1, discount_factor=0.95, epsilon=0.3)
    env = TicTacToeEnv()
    random_opp = RandomAgent()

    # quick training: 5000 episodes of self-play
    for _ in range(5000):
        state = env.reset()
        while not env.done:
            valid = env.get_valid_actions()
            action = agent.choose_action(state, valid, training=True)
            next_state, reward, done, info = env.step(action)

            if done:
                agent.update(state, action, reward, next_state, [], done=True)
            else:
                agent.update(state, action, 0.0, next_state,
                             env.get_valid_actions(), done=False)
            state = next_state
        agent.decay_epsilon()

    # evaluate: play 200 games as X against random
    wins = 0
    for _ in range(200):
        state = env.reset()
        while not env.done:
            valid = env.get_valid_actions()
            if env.current_player == 1:
                action = agent.choose_action(state, valid, training=False)
            else:
                action = random_opp.choose_action(state, valid)
            state, _, done, info = env.step(action)

        if info["winner"] == 1:
            wins += 1

    win_rate = wins / 200
    print(f"\nQuick test win rate: {win_rate:.1%}")
    assert win_rate > 0.5, f"Agent should win > 50% but got {win_rate:.1%}"


if __name__ == "__main__":
    test_initial_board_is_empty()
    test_valid_actions_decreases()
    test_player_alternates()
    test_win_detection()
    test_draw_detection()
    test_invalid_move_raises()
    test_random_agent_always_picks_valid()
    test_q_agent_learns_something()
    print("\n✅ All tests passed!")
