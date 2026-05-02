

Robust Reinforcement Learning means to train without very costly mistakes. In the context of a robot learning to drive a car it means to 
avoid fatal crashes, because crashing requires to purchase a new car. In a environment like this SARSA can be omre catious than Q-Learning, 
which I tested with the cliff walk. Below are the outputs for 10% wind chance:

--- Q-Learning ---

Finished in 1.36 seconds

Policy Value: 5.3864

Q-Learning Policy:
------------------------------------------------------------
| ↓  | B  | B  | B  | B  | B  | B  | B  | B  | B  | B  | G  |
------------------------------------------------------------
| ↓  | ←  | →  | →  | →  | ↓  | ←  | →  | →  | ↓  | →  | ↑  |
------------------------------------------------------------
| ↓  | ↓  | ↓  | ↓  | →  | ↓  | →  | →  | →  | →  | →  | ↑  |
------------------------------------------------------------
| →  | →  | →  | →  | →  | →  | →  | →  | →  | →  | →  | ↑  |
------------------------------------------------------------
--- SARSA ---

Finished in 1.35 seconds

Policy Value: 6.0025

SARSA Policy:
------------------------------------------------------------
| ↓  | B  | B  | B  | B  | B  | B  | B  | B  | B  | B  | G  |
------------------------------------------------------------
| ↓  | ↓  | ↓  | ↓  | →  | ↓  | ↓  | ↓  | ↓  | →  | →  | ↑  |
------------------------------------------------------------
| ↓  | ↓  | ↓  | ↓  | →  | →  | →  | →  | →  | →  | →  | ↑  |
------------------------------------------------------------
| →  | →  | →  | →  | →  | →  | →  | →  | →  | →  | →  | ↑  |
------------------------------------------------------------


As one can see SARSA gets a higher policy value and the arrows show that it takes the safer path. Q_learning still sometimes walks along the edge of the cliff.
The differences come from the calculation of the respective update rule. In Q-Learning, when calculating the future value it uses the max operator, meaning it 
assumes that it will play perfectly in the next state. While SARSA on the other hand, uses the actual next action. So it includes the epsilon chance of making 
a random mistake. Q-learning only evaluates the states based on the environment, while SARSA does it based on the actual epsilon-greedy policy. That makes SARSA
more risk averse. Below is a test with 0% wind:


--- Q-Learning ---

Finished in 0.91 seconds

Policy Value: 21.0672

Q-Learning Policy:
------------------------------------------------------------
| ↓  | B  | B  | B  | B  | B  | B  | B  | B  | B  | B  | G  |
------------------------------------------------------------
| →  | →  | →  | →  | →  | →  | →  | →  | →  | →  | →  | ↑  |
------------------------------------------------------------
| ↑  | →  | ↑  | ↑  | ↑  | ↑  | →  | →  | →  | ↑  | ↑  | ↑  |
------------------------------------------------------------
| ↑  | →  | ↑  | ↑  | ↑  | ↑  | ←  | →  | ↑  | ↑  | ↑  | ↑  |
------------------------------------------------------------
--- SARSA ---

Finished in 1.13 seconds

Policy Value: 15.1645

SARSA Policy:
------------------------------------------------------------
| ↓  | B  | B  | B  | B  | B  | B  | B  | B  | B  | B  | G  |
------------------------------------------------------------
| ↓  | ↓  | ↓  | ↓  | ↓  | ↓  | ↓  | ↓  | ↓  | →  | →  | ↑  |
------------------------------------------------------------
| →  | →  | →  | →  | →  | →  | →  | →  | →  | →  | →  | ↑  |
------------------------------------------------------------
| →  | →  | ↑  | ↑  | →  | →  | ↑  | ↑  | ↑  | ↑  | ↑  | ↑  |
------------------------------------------------------------

Even here SARSE doesnt walk next to the cliff, because it knows it could make a mistake. Meanwhile Q learning just walks along the edge, even if it could fall.
That results in a higher score for Q-learning become it plays optimally considering there is no wind. That also means that during training it will fall off the cliff.
In the scenario described above that would result in high costs during training. That doesn't happen with SARSA. Finally that means that Q-learning has a better performance
for the final polished policy, while SARSA has the superior performance during the training phase.






