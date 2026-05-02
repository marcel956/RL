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
from game_dynamic_algorithms import value_iteration


# ==========================================
# 1. Setup the Environment
# ==========================================
# We create a simple grid where a high reward is far away, 
# and a small penalty applies to every step.
rewards = {
    (0, 3): {"type": "goal", "reward_type": "deterministic", "value": 1, "is_terminal": True},
    (1, 3): {"type": "bomb", "reward_type": "deterministic", "value": -1, "is_terminal": True}
}

noise_dirs = {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25}

env = gridworld(
    m=5, n=5, 
    reward_structure=rewards, 
    default_reward=-0.04,  # Small step penalty to encourage efficiency
    wall_behavior="reflect", 
    start_state=(4, 0), 
    wind_direction="up", 
    wind_prob=0.0,  # Keeping dynamics simple for clear value visualization
    slip_prob=0.0, 
    noise_prob=0.2, 
    noise_directions=noise_dirs
)

def plot_policy(env, policy, title):
    """
    Plottet die Arrows der Policy und die Belohnungen der Terminals 
    basierend auf der spezifischen gridworld-Klasse.
    """
    # Mapping deiner String-Aktionen auf Symbole
    arrows = {
        'up': '↑',
        'down': '↓',
        'left': '←',
        'right': '→'
    }

    # Dynamische Größe basierend auf der Grid-Größe (n=Breite, m=Höhe)
    fig, ax = plt.subplots(figsize=(env.n, env.m))
    
    # Grenzen setzen: x geht von 0 bis n, y von 0 bis m
    # Wir invertieren den y-Limit, damit Zeile 0 (wie im Array) oben ist
    ax.set_xlim(-0.5, env.n - 0.5)
    ax.set_ylim(env.m - 0.5, -0.5)
    
    ax.set_xticks(range(env.n))
    ax.set_yticks(range(env.m))
    ax.grid(True, linestyle='--', color='gray')
    
    for r in range(env.m):
        for c in range(env.n):
            state = (r, c)
            
            if state in env.terminal_states:
                # Terminal-Zustände: Zeige die erwartete Belohnung
                # Wir holen den Wert aus dem vorberechneten expected_rewards Matrix
                rew = env.expected_rewards[r, c]
                color = 'green' if rew > 0 else 'red'
                ax.text(c, r, f"{rew:+.1f}", ha='center', va='center', 
                        weight='bold', fontsize=14, color=color)
            
            elif state in policy:
                # Normale Zustände: Zeichne den Aktions-Pfeil
                action = policy[state]
                # Falls die Policy eine Liste ist (stochastisch), zeigen wir ein '?'
                symbol = arrows.get(action, '?')
                ax.text(c, r, symbol, ha='center', va='center', fontsize=22)

    ax.set_title(title, fontsize=14, pad=10)
    plt.tight_layout()
    plt.show()



finite_val_iter_3 = value_iteration(env, gamma=1.0, V_star=None, epsilon=-1, max_steps=3, async_update=False, use_Q=False)
finite_val_iter_10 = value_iteration(env, gamma=1.0, V_star=None, epsilon=-1, max_steps=10, async_update=False, use_Q=False)
finite_val_iter_20 = value_iteration(env, gamma=1.0, V_star=None, epsilon=-1, max_steps=20, async_update=False, use_Q=False)

infinite_val_iter = value_iteration(env, gamma=0.5, V_star=None, epsilon=1e-6, max_steps=None, async_update=False, use_Q=False)
infinite_val_iter_2 = value_iteration(env, gamma=0.05, V_star=None, epsilon=1e-6, max_steps=None, async_update=False, use_Q=False)

plot_policy(env, finite_val_iter_3[1], 'Finite Value Iteration: gamma = 1.0, max iter =  3')
plot_policy(env, finite_val_iter_10[1], 'Finite Value Iteration: gamma = 1.0, max iter = 10')
plot_policy(env, finite_val_iter_20[1], 'Finite Value Iteration: gamma = 1.0, max iter = 20')
plot_policy(env, infinite_val_iter[1], 'Infinite Value Iteration: gamma = 0.5')
plot_policy(env, infinite_val_iter_2[1], 'Infinite Value Iteration: gamma = 0.05')


## Comment zu 4b:
# MDPs mit endlicher Laufzeit dauern nicht unendlich lange und können als episodisch betrachtet werden; so hat beispielsweise eine Schachpartie ein Ende, 
# während eine andere Aufgabe, wie die eines Sicherheitsroboters, möglicherweise keine Endzeit hat. Für den Fall der unendlichen Laufzeit verwenden wir 
# diskontierte Belohnungen, um die Konvergenz zu einem Fixpunkt sicherzustellen. Die Strategie und die gewählten Aktionen können sich in einer Umgebung mit 
# endlicher Zeit aufgrund des zusätzlichen Zeitdrucks ebenfalls ändern. Das bedeutet, dass wir für unendliche MDPs eine stationäre optimale Strategie finden können, 
# für endliche jedoch nicht, da die verbleibende Zeit einen Einfluss hat. Wenn beispielsweise nur noch wenige Runden übrig sind, kann die Strategie aggressiver sein, um den Endzustand zu erreichen und die Belohnung zu erhalten.