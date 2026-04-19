## Task 5:




## Task 6:

Green converges fastest to the optimal policy, seen in the RMSE, while red and blue dont converge as fast, red even flatens out. Thats because its learning rate of 1/n gets too low
too make real progress. 
For the Backpropagation effect the green one also looks like the best. it converges nicely to 6 which should be the real optimal value of the start state. Meanwhile the other two climb very slowly
towards it. The constant schedule will probably also reach it after more steps while the 1/n rate gets stuck at 50% of the correct value. Once again because it step size gets so low that it barely makes progress.
In theory 1/n is good and used in proofs, but in practice it has its shortcomings of getting too small too fast. So even if it would converge to the correct value in infinite time, a stepsize
with a slower decay is better, like 1/sqrt(n).

