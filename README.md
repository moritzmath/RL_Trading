# Q-Learning Agent for Stock Trading

This project implements a Q-Learning reinforcement learning agent that learns to trade Apple (AAPL) stock based on technical indicators. The goal is to evaluate whether a tabular Q-Learning strategy can outperform the asset itself and random strategies.

## Overview

This repository includes:

-  A custom OpenAI Gym-like trading environment
-  A Q-Learning agent with epsilon-greedy exploration strategy
-  Technical indicators as state features: Simple Moving Average (SMA), Exponential Moving Average (EMA), Moving Average Crossover, Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Bollinger Bands (BB), Stochastic Oscillator (SO)
- Evaluation against baseline strategies (buy-and-hold and random)

## Project Structure
- q_learning_trading.ipynb: Main notebook for training & evaluation
- trading_environment.py: Custom trading environment (Gym-style)
- q_learning_agent.py: Q-Learning agent
- technical_indicators.py: File for computation of the technical indicators

## Package Requirements  
To use this library, make sure you have the following packages installed:
* [Numpy](https://numpy.org)
* [Pandas](https://pandas.pydata.org)
* [Matplotlib](https://matplotlib.org)
* [Gymnasium](https://gymnasium.farama.org)
* [yfinance](https://pypi.org/project/yfinance/)

## Results & Evaluation
After training, the Q-Learning agent is tested on unseen data and evaluated based on:
- Final portfolio value
- Cumulative returns
- Comparison to:
  - A buy-and-hold strategy
  - A randomly acting agent
    
All metrics are visualized in the notebook.

## Methodology
The agent receives a discrete state space derived from technical indicator signals and selects a discrete trading action (stock position). The reward is based on the log change in portfolio value. Training is performed over multiple episodes with an epsilon-greedy strategy for exploration.

## Notes
- The action represents how many shares to hold, not how many to buy/sell.
- The environment includes transaction costs and explicitly tracks cash and position.
- The notebook supports easy comparison between Q-Learning, random, and static strategies.

