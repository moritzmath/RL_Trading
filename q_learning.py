import numpy as np
import random
from collections import defaultdict

seed = 42
random.seed(seed)


class QLearningAgent():
    """
    Q-Learning agent for our Trading Environment
    """

    def __init__(self, action_space, observation_space, alpha=0.1, gamma=0.99, epsilon=1, epsilon_min=0.01, epsilon_decay=0.99):
        """
        Initializes the Q-learning agent with given hyperparameters.

        Args:
            action_space (gym.Space): The action space.
            observation_space (gym.Space): The observation space.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial exploration rate.
            epsilon_min (float): Minimum exploration rate.
            epsilon_decay (float): Decay rate of exploration.
        """

        self.action_space=action_space
        self.observation_space=observation_space

        self.q_table = defaultdict(lambda: np.zeros(action_space.n))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state (array-like): Current state of the environment.

        Returns:
            int: Chosen action.
        """

        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[tuple(state)]) 
        
    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Bellman equation.

        Args:
            state (array-like): Previous state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (array-like): Resulting state.
        """

        state = tuple(state)
        next_state = tuple(next_state)

        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action]  += self.alpha * td_error

    def train(self, env, num_episodes=500):
        """
        Trains the agent using the Q-learning algorithm.

        Args:
            env (gym.Env): The trading environment.
            num_episodes (int): Number of episodes to train the agent for.
        """

        for episode in range(num_episodes):
            state, _ = env.reset(seed=seed)
            done = False

            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                self.update_q_table(state, action, reward, next_state)
                state = next_state

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            print(f"Episode {episode + 1}/{num_episodes}, Epsilon: {self.epsilon:.3f}", end='\r', flush=True)
        print()
        print('Training finished!')


