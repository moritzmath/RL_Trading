import gymnasium as gym 
import numpy as np
import pandas as pd
from gymnasium.spaces import Discrete


class TradingEnv(gym.Env):
    """
    A custom Gym environment for simulating trading using discrete technical indicators.
    """

    def __init__(self, data, initial_cash=10000, max_num_shares=10, max_short_position=0, cost_per_trade=0.1):
        super(TradingEnv, self).__init__()
        """"
        Initialize the trading environment.

        Args:
            data (pd.DataFrame): Preprocessed market data with technical indicators.
            initial_cash (float): Starting cash balance for the agent.
            max_num_shares (int): Maximum number of shares the agent is allowed to hold.
            max_short_position (int): Maximum number of shares the agent is allowed to short.
            cost_per_trade (float): Transaction cost per trade unit.
        """
        
 
        self.data = data
        self.max_num_shares = max_num_shares
        self.max_short_position = max_short_position
        self.current_step = 0
        self.cost_per_trade = cost_per_trade

        self.action_space = gym.spaces.Discrete(max_num_shares + max_short_position + 1, start=-max_short_position)

        self.observation_space = gym.spaces.Tuple((Discrete(3, start=-1), 
                                                  Discrete(3, start=-1), 
                                                  Discrete(3, start=-1), 
                                                  Discrete(3, start=-1), 
                                                  Discrete(3, start=-1),
                                                  Discrete(max_num_shares + max_short_position + 1, start=-max_short_position)), 
                                                  seed=42
                                                )

        self.initial_cash = initial_cash
        self.cash = initial_cash

        self.stock_value = 0
        self.current_position = 0

        self.portfolio_value = self.initial_cash + self.stock_value

        self.current_open_price = self.data.iloc[self.current_step]['Open']
        self.current_close_price = self.data.iloc[self.current_step]['Close']


    def reset(self, seed=None):
        """
        Reset the environment to the initial state.

        Args:
            seed (int, optional): Seed for reproducibility.

        Returns:
            tuple: Initial observation and an empty info dictionary.
        """
        super().reset(seed=seed)

        self.current_step = 0

        self.cash = self.initial_cash
        self.stock_value = 0
        self.current_position = 0

        self.portfolio_value = self.cash 

        self.current_open_price = self.data.iloc[self.current_step]['Open']
        self.current_close_price = self.data.iloc[self.current_step]['Close']

        return self.get_observation(), {}
    
    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action (int): Number of shares to hold (can be negative for shorting).

        Returns:
            tuple: (next_state, reward, done, info)
        """
        

        self.current_open_price = self.data.iloc[self.current_step]['Open']
        self.current_close_price = self.data.iloc[self.current_step]['Close']

        transaction_cost = abs(action - self.current_position) * self.cost_per_trade

        new_cash = self.cash + (self.current_position - action) * self.current_open_price - transaction_cost
        new_stock_value = action * self.current_close_price
        new_portfolio_value = new_cash + new_stock_value
        

        reward = np.log(new_portfolio_value / self.portfolio_value)

        self.current_step += 1

        done = self.current_step >= len(self.data) - 1

        self.current_position = action
        self.cash = new_cash
        self.stock_value = new_stock_value
        self.portfolio_value = new_portfolio_value

        next_state = self.get_observation()

        return next_state, reward, done, {}

    def get_observation(self):
        """
        Get the current observation.

        Returns:
            np.ndarray: Array of discretized technical indicators and position.
        """
        
        return np.array([self.data.iloc[self.current_step]['Crossover_signal'], 
                         self.data.iloc[self.current_step]['RSI_signal'], 
                         self.data.iloc[self.current_step]['MACD_signal'], 
                         self.data.iloc[self.current_step]['BB_signal'], 
                         self.data.iloc[self.current_step]['SO_signal'], 
                         self.current_position
                         ], dtype=np.int32)

