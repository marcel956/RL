## Exercise 5
Backpropagation in this context is that the algorithm learns backwards. In the first step it figures out that the points adjacent to the goal get the most reward if they step
on the goal. In the step after it learns to go to the points adjacent to the goal because they will earn reward after. So it works step by step towards the beginning, after 
which it then found the optimal path to take.
The stochastic control part comes into play when you introduce randomness into the game, like wind etc. Now the optimal decision isn't just a single path. It calculates a
robust safety net so that it still knows what to do if randomness comes into play.
Policy evaluations teaches us that we don't need to play the game at all to figure out what a policy will score on average. We just pass the theoretical points backwards from
goal to start. With the transition probabilities we can then get the average score of a policy.


## Exercise 7
??

## Exercise 8
I plotted the difference of the finite and infinite agent.
The shortest possible path to the goal is 6 steps, so all graphs end there. If T<6 the agent doesn't have enough time to reach the goal. But after that the finite and infinite
agents both align.
The Minus environment (green) has the steepest line and drops the fastest. Both finite and infinite have the same strategy to reach the goal asap, because they are bleeding -1s. 
So the difference is the smallest and shrinks the fastest.
In the Zero environment (blue) there is no penalty for taking your time for the infinite agent. But also no reward. So the difference shrinks steadily by the discount factor gamma.
In The Plus environment (orange) the line stays flat until T=6. The infinite agent knows it has forever so it just wanders around and picks up +1s. The finite agent when T<6 knows  it 
cant reach the goal so it just takes a few steps. For T=6 both reach the goal, because wandering around isn't worth with the discount factor. Then the difference is zero

