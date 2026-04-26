import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

sheet4_path = Path(__file__).parent.parent / "Sheet4"
sheet5_path = Path(__file__).parent.parent / "Sheet5"
sheet6_path = Path(__file__).parent.parent / "Sheet6"



# Add this folder to Python's search path
sys.path.append(str(sheet4_path))
sys.path.append(str(sheet5_path))


from gridworld import gridworld
from hard_policy_evaluation import policy_evaluation, value_iteration, monte_carlo_optimal_policy, worst_value_iteration
from game_dynamic_algorithms import policy_iteration, value_iteration, policy_evaluation
from dynamic_programming import policy_evaluation_finiteMDP, optimal_control
from sample_based_algorithms import monte_carlo_Q, monte_carlo_V, totally_async_policy_evaluation, Q_learning, RMSE_evaluation, Q_into_policy, Q_into_V, evaluate_pit_stop