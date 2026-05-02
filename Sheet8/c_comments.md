## Backpropagation

Backpropagation in this context is that the algorithm learns backwards. In the first step it figures out that the points adjacent to the goal get the most reward if they step
on the goal. In the step after, it learns to go to the points adjacent to the goal because they will earn reward after that. So it works step by step towards the beginning, after 
which it then found the optimal path to take.
The stochastic control part comes into play when you introduce randomness into the game, like wind etc. Now the optimal decision isn't just a single path. It calculates a
robust safety net so that it still knows what to do if randomness comes into play.
Policy evaluations teaches us that we don't need to play the game at all to figure out what a policy will score on average. We just pass the theoretical points backwards from
goal to start. With the transition probabilities we can then get the average score of a policy. See picture outputs of the code.



## Robust Reinforcement Learning

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

Even here SARSA doesn't walk next to the cliff, because it knows it could make a mistake. Meanwhile Q learning just walks along the edge, even if it could fall.
That results in a higher score for Q-learning become it plays optimally considering there is no wind. That also means that during training it will fall off the cliff.
In the scenario described above that would result in high costs during training. That doesn't happen with SARSA. Finally that means that Q-learning has a better performance
for the final polished policy, while SARSA has the superior performance during the training phase.



## Overestimation Bias

The overestimation bias is a flaw of Q-learning that comes from it using max to update the Q-values. For random rewards the max operator can skew towards positive results. 
So if it gets lucky and a good positive reward once, it needs to get unlucky after that to train that bias out. To test this i used a small 3x4 gridworld environment with
a goal in the top left and a few casino fields on the right. The casinos randomly give you either reward -15 or 15. I tested Q-learning against SARSA since that doesn't sample 
the theoretical maximum. I calculated the bias and squared bias as well.
Just by looking at the biases you can see that SARSA performs better. Both it's biases are much smaller than for Q-learning. Q-learning actually gets a squared total bias of
100 while its total bias is -16. Even though the total bias is -16, overestimation is still shown. The negative values come from the flaw of initializing the q-table at 0.
If the agent steps onto the casino and get a negative reward on the first try, the Q value drops below zero and it will avoid it. Causing it to not update that Q-value.
On the other hand if the casino gives a positive number, it will step on it again until it got unlucky enough times. The overestimation bled backwards into the safe states,
for example (1, 1) | down and (1, 0) | right.
This also shows that the bias is not uniform. If everything was overestimated by +2 it wouldn't be a problem since the best action would still be the best action.
But here the casino exits have high negative bias (-8.57) while the safe states have a positive bias (0.17). Thats why the squared bias is a good tool to measure 
the amount of delusion the agent has.



==================================================
--- Q-Learning ---
==================================================

Q-Learning Bias Metrics:
State      | Action     | Estimated Q  | True Q     | Bias
------------------------------------------------------------
(0, 1)     | down       | 0.8626       | 0.8100     | 0.0526
(0, 1)     | left       | 1.0000       | 1.0000     | 0.0000
(0, 1)     | right      | -1.3386      | 0.0000     | -1.3386
(0, 3)     | down       | 0.0000       | 0.0000     | 0.0000
(0, 3)     | left       | 0.0000       | 0.0000     | 0.0000
(1, 0)     | up         | 1.0000       | 1.0000     | 0.0000
(1, 0)     | down       | 0.7281       | 0.6561     | 0.0720
(1, 0)     | right      | 0.8758       | 0.8100     | 0.0658
(1, 1)     | up         | 0.9079       | 0.9000     | 0.0079
(1, 1)     | down       | 0.9078       | 0.7290     | 0.1788
(1, 1)     | left       | 0.8956       | 0.9000     | -0.0044
(1, 1)     | right      | 0.8447       | 0.7290     | 0.1157
(1, 2)     | up         | -4.5000      | 0.0000     | -4.5000
(1, 2)     | down       | -8.5714      | 0.0000     | -8.5714
(1, 2)     | left       | 0.8640       | 0.8100     | 0.0540
(1, 2)     | right      | -1.8750      | 0.0000     | -1.8750
(2, 0)     | up         | 0.8242       | 0.9000     | -0.0758
(2, 0)     | right      | 0.8921       | 0.7290     | 0.1631
(2, 1)     | up         | 0.9074       | 0.8100     | 0.0974
(2, 1)     | left       | 0.8476       | 0.6561     | 0.1915
(2, 1)     | right      | -1.1628      | 0.0000     | -1.1628
(2, 3)     | up         | 0.0000       | 0.0000     | 0.0000
(2, 3)     | left       | 0.0000       | 0.0000     | 0.0000
------------------------------------------------------------
Summed Total Bias:         -16.5292
Summed Squared Total Bias: 100.5181

Q-Learning Final Policy:
--------------------
| G  | ←  | B  | ↓  |
--------------------
| ↑  | ↑  | ←  | B  |
--------------------
| →  | ↑  | B  | ↑  |
--------------------

==================================================
--- SARSA ---
==================================================

SARSA Bias Metrics:
State      | Action     | Estimated Q  | True Q     | Bias
------------------------------------------------------------
(0, 1)     | down       | 0.3267       | 0.0000     | 0.3267
(0, 1)     | left       | 0.1456       | 1.0000     | -0.8544
(0, 1)     | right      | 0.0038       | 0.0000     | 0.0038
(0, 3)     | down       | 0.0000       | 0.0000     | 0.0000
(0, 3)     | left       | 0.0000       | 0.0000     | 0.0000
(1, 0)     | up         | 1.0000       | 1.0000     | 0.0000
(1, 0)     | down       | 0.1280       | 0.8100     | -0.6820
(1, 0)     | right      | 0.0588       | 0.0000     | 0.0588
(1, 1)     | up         | 0.0558       | 0.0000     | 0.0558
(1, 1)     | down       | 0.1184       | 0.0000     | 0.1184
(1, 1)     | left       | 0.1621       | 0.9000     | -0.7379
(1, 1)     | right      | 0.3201       | 0.0000     | 0.3201
(1, 2)     | up         | 0.0266       | 0.0000     | 0.0266
(1, 2)     | down       | 0.0836       | 0.0000     | 0.0836
(1, 2)     | left       | 0.3297       | 0.0000     | 0.3297
(1, 2)     | right      | -0.5516      | 0.0000     | -0.5516
(2, 0)     | up         | 0.8411       | 0.9000     | -0.0589
(2, 0)     | right      | 0.3235       | 0.0000     | 0.3235
(2, 1)     | up         | 0.3335       | 0.0000     | 0.3335
(2, 1)     | left       | 0.1246       | 0.8100     | -0.6854
(2, 1)     | right      | -0.5397      | 0.0000     | -0.5397
(2, 3)     | up         | 0.0000       | 0.0000     | 0.0000
(2, 3)     | left       | 0.0000       | 0.0000     | 0.0000
------------------------------------------------------------
Summed Total Bias:         -2.1296
Summed Squared Total Bias: 3.3704

SARSA Final Policy:
--------------------
| G  | ↓  | B  | ↓  |
--------------------
| ↑  | →  | ←  | B  |
--------------------
| ↑  | ↑  | B  | ↑  |
--------------------