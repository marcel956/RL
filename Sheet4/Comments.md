## Exercise 6
# Complexity: 
Dynamic Programming/Value Iteration checks every action for every state, and for every action every possible
next state, so O(|S| * |A| * |S|). But one needs to know the exact transition model. The algorithm scales with size
of the board/game.
Monte Carlo:
Complexity is O(N*L) with N the number of episodes and L the maximum length of one. You don't need to know the full
board because every episode only simulates one path. But N needs to be large to average out the randomness to get a good estimate.

# Worst-Case Environments
For Monte Carlo imagine a 1000x1000 Gridworld board with every tile being 0 reward and one being +100. The board is so huge 
that the probability of landing on the goal is basically zero. So Monte Carlo randomly wonders around not learning anything,
because it never stepped on the reward.
For Value Iteration imagine a bandit tree with 100 branches per step and 10 steps per game. The number of states is 100^10. 
This scenario describes the Curse of Dimensionality. The Algorithm will try to build a dictionary with 100^10 keys, which is
so large that the Computer/Python won't be able to handle it and crash. Meanwhile Monte Carlo could at least sample a few paths
and give you an estimate.

# Human vs. Algorithm
A Human can solve a Gridworld map basically instantaneous. You will look at the map see the goal and start and instantly draw
a map. Every corner thats uninteresting will be ignored because you intuitively know its useless. So for small puzzles the time
to solve is O(1).
Monte Carlo cant to that, it just goes by trial and error. So it bashes into every wall and tries out everything and fails, until 
the statistical average of the failures points to the optimal path.
It's similar for the value iteration algorithm. It just tries out every possible path to prove that every corner of the map is a 
bad idea. It blindly passes numbers from goal to the start step-by-step until finding the optimal route.

Algorithms are only better than humans if the problem because so complex that a human can't solve it in their head anymore.


## Exercise 7

Best policies: When looking at those, the wind environment has the best score. That's because the wind is predictable, it
always blows to the right. So the agent learns to use it to its advantage. For the noise environment the score gets worse
the agent can't use it to an advantage so it takes longer to reach the goal, resulting in a lower score. The slip environment
is the worst because walking along the bomb can push you into it quickly, so the agent has to take routes along the edges to 
not get pushed into it.

Worst policies: Not much difference in the environments. The bomb is close to the start so it only takes two steps to get there.
It gets reached quickly in every environment, so the score for it is always -10.

Random and Uniformly random: They both perform very bad in every environment. Which makes sense because both policies are just 
blindly guessing a path. So the score are at around -7 to -9. Sometimes fixed random can get lucky and constantly bump into a
wall, which results in a lower score because it avoids the bomb.