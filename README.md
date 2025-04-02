# RL_Trading


My attempt at implementing a reinforcement learning agent for trading. Inputs: 
- *Max_num_shares* = Maximal number of stocks we are allowed to hold at a time
- *cost_per_trade* = Transaction cost per executed trade

Action $a_t$ is to be understood as the number of stocks held at time $t$. Using this, we conclude that the transaction cost can be calculated by
$ |a_t - a_{t-1}| * cost_per_trade $
