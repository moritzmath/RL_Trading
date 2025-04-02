# RL_Trading


My attempt at implementing a reinforcement learning agent for trading. Inputs: 
- *Max_num_shares* = Maximal number of stocks we are allowed to hold at a time
- *cost_per_trade* = Transaction cost per executed trade, denoted by c

Action $a_t$ is to be understood as the number of stocks held at time $t$. Using this, we conclude that the transaction cost can be calculated by
$c_t = |a_t - a_{t-1}| * c$.
Hence, the total profit evolves according to the following recursive formula:
$p_t = p_{t-1} + a_t * (op_t - cp_t) - c_t$
where $op_t$ denotes the opening price at time $t$ and $cp_t$ denotes the closing price at time $t$.
