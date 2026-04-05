## Exercise 5
Backpropagation in this context is that the algorithm learns backwards. In the first step it figures out that the points adjacent to the goal get the most reward if they step
on the goal. In the step after it learns to go to the points adjacent to the goal because they will earn reward after. So it works step by step towards the beginning, after 
which it then found the optimal path to take.
The stochastic control part comes into play when you introduce randomness into the game, like wind etc. Now the optimal decision isn't just a single path. It calculates a
robust safety net so that it still knows what to do if randomness comes into play.
Policy evaluations teaches us that we don't need to play the game at all to figure out what a policy will score on average. We just pass the theoretical points backwards from
goal to start. With the transition probabilities we can then get the average score of a policy.