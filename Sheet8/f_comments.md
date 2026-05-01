
## No noise environment optimization:

# First Test
Below is the output from the first test with epsilon = 0.05, num_episodes=5000, schedule= "constant". As one can see here both
fall for the fake goal trap. They don't have the exact same policy but the value is the same because they move the same from the start to the fake goal.

--- Standard Q-Learning ---

Finished in 0.15 seconds

Policy Value: 0.5265

Standard Q-Learning Policy:
--------------------
| G  | ←  | ←  | ←  |
--------------------
| ↑  | ↑  | ↑  | ←  |
--------------------
| ↑  | ↑  | ↑  | ↑  |
--------------------
| ↑  | G  | ←  | ↑  |
--------------------

--- Double Q-Learning ---

Finished in 0.15 seconds

Policy Value: 0.5265

Double Q-Learning Policy:
--------------------
| G  | ←  | ←  | ←  |
--------------------
| ↑  | ↑  | ←  | ↑  |
--------------------
| ↑  | ↑  | ↑  | ↑  |
--------------------
| ↑  | G  | ↑  | ↑  |
--------------------

# Second Test
For the next test I choose epsilon to be 0.5. So now the algorithm explores half the time. This lead to DoubleQ sometimes (but rarely)
finding the goal and having a the optimal policy that walks straight to it. 
Meanwhile StandardQ gets stuck in the stochastic region, probably because it received some good rewards at the beginning.


--- Standard Q-Learning ---

Finished in 0.71 seconds

Policy Value: -0.4500

Standard Q-Learning Policy:
--------------------
| G  | ↓  | →  | ↓  |
--------------------
| →  | ↓  | →  | ↓  |
--------------------
| →  | ↓  | ←  | ↓  |
--------------------
| →  | G  | ←  | ↑  |
--------------------

--- Double Q-Learning ---

Finished in 0.40 seconds

Policy Value: 0.6561

Double Q-Learning Policy:
--------------------
| G  | ↓  | ←  | ←  |
--------------------
| ↑  | ↓  | ↑  | ↑  |
--------------------
| ↓  | ↓  | ←  | ←  |
--------------------
| →  | G  | ←  | ←  |
--------------------

# Third Test
Next I tested slightly higher epsilon values till 0.25 which all lead to the same outcome as in 1. After that I upped the number of episodes to 15000, 
with epsilon 0.05 and 0.1 it still had the same results as in 1. Epsilon 0.15 then found the right policy occasionally, but only for StandardQ. 
Epsilon 0.2 found it reliably for StandardQ and sometimes for DoubleQ. They trend seems to follow for even higher epsilon values. This run below
had epsilon on 0.3 and 25000 episodes. DoubleQ still couldn't find the optimal path reliably

--- Standard Q-Learning ---

Finished in 1.97 seconds

Policy Value: 0.6561

Standard Q-Learning Policy:
--------------------
| G  | ↓  | ↓  | ↓  |
--------------------
| ↓  | ↓  | ←  | ←  |
--------------------
| ↓  | ↓  | ←  | ↑  |
--------------------
| →  | G  | ←  | ←  |
--------------------

--- Double Q-Learning ---

Finished in 1.27 seconds

Policy Value: 0.5265

Double Q-Learning Policy:
--------------------
| G  | ←  | ←  | ←  |
--------------------
| ↑  | ↓  | ↑  | ↑  |
--------------------
| →  | ↓  | ←  | ↑  |
--------------------
| ↑  | G  | ←  | ↑  |
--------------------


# Forth Test
Next I tested with epsilon 0.25 and 15000 episodes and 1/n and 1/sqrt(n) as schedule for step size. This didn't really change anything 
in the results.

--- Standard Q-Learning ---

Finished in 1.08 seconds

Policy Value: 0.6561

Standard Q-Learning Policy:
--------------------
| G  | ↓  | ↓  | ↓  |
--------------------
| →  | ↓  | ←  | ←  |
--------------------
| →  | ↓  | ←  | ↑  |
--------------------
| →  | G  | ←  | ←  |
--------------------

--- Double Q-Learning ---

Finished in 0.71 seconds

Policy Value: 0.5265

Double Q-Learning Policy:
--------------------
| G  | ←  | ←  | ←  |
--------------------
| ↑  | ↑  | ↑  | ↑  |
--------------------
| →  | ↓  | ←  | ↑  |
--------------------
| ↑  | G  | ↑  | ←  |
--------------------


# Fifth Test
At the end I did a Parameter Sweep to get a bigger picture. All in all Double Q seems more inconsistent than the Standard version, thats because it's very sample inefficient.
Having two Q tables means you need twice the samples to update both. So it needs more samples and/or a higher exploration rate to avoid falling for the fake goal.
So even if its chance of landing in the stochastic zone is smaller, it seems worse.
Meanwhile StandardQ was vulnerable to falling step size schedules and sometimes landed in the stochastic zone with them. All in all it still found the real goal more often.

============================================================
 Parameter Sweep Results 
============================================================
Epsilon    | Schedule     | Standard Q Value   | Double Q Value    
-----------------------------------------------------------------
0.10       | constant     | 0.6561             | 0.5265            
0.10       | 1/n          | 0.5706             | 0.5265            
0.10       | 1/sqrt(n)    | 0.6561             | 0.5265            
0.20       | constant     | 0.6561             | 0.5265            
0.20       | 1/n          | 0.5341             | -0.2368           
0.20       | 1/sqrt(n)    | 0.6561             | 0.5265            
0.30       | constant     | 0.0000             | 0.5265            
0.30       | 1/n          | -0.4500            | 0.5265            
0.30       | 1/sqrt(n)    | 0.6561             | 0.5265            
0.40       | constant     | 0.6561             | 0.5265            
0.40       | 1/n          | 0.6561             | 0.6561            
0.40       | 1/sqrt(n)    | 0.6561             | 0.6561            
============================================================



## Environment with 20% Chance of random noise

Here I just ran a bigger Parameter Sweep. One can see fast that the score drops across the board. Reaching the goal in 5 steps becomes way harder because of the noise,
thats why the max value of 0.6561 is not getting reached anymore. StandardQ has the higher highs and is usually better than DoubleQ. But there are cases where StandardQ
gets a really bad value like at 0.25 with 1/n, so DoubleQ seems to be more consistent in that regard.



============================================================
 Parameter Sweep Results 
============================================================
Epsilon    | Schedule     | Standard Q Value   | Double Q Value    
-----------------------------------------------------------------
0.05       | constant     | 0.4790             | 0.4864            
0.05       | 1/n          | 0.5771             | 0.4900            
0.05       | 1/sqrt(n)    | 0.5779             | 0.4857            
0.10       | constant     | 0.4920             | 0.4958            
0.10       | 1/n          | 0.5783             | 0.5036            
0.10       | 1/sqrt(n)    | 0.5767             | 0.4896            
0.15       | constant     | 0.4731             | 0.4821            
0.15       | 1/n          | 0.5692             | 0.5054            
0.15       | 1/sqrt(n)    | 0.5780             | 0.4940            
0.20       | constant     | 0.1431             | 0.4895            
0.20       | 1/n          | 0.5768             | 0.5108            
0.20       | 1/sqrt(n)    | 0.5739             | 0.4940            
0.25       | constant     | 0.4916             | 0.4940            
0.25       | 1/n          | 0.1026             | 0.5138            
0.25       | 1/sqrt(n)    | 0.5766             | 0.4896            
0.30       | constant     | 0.5144             | 0.4938            
0.30       | 1/n          | 0.5735             | 0.5142            
0.30       | 1/sqrt(n)    | 0.5767             | 0.4858            
0.35       | constant     | 0.1449             | 0.4985            
0.35       | 1/n          | 0.5784             | 0.4865            
0.35       | 1/sqrt(n)    | 0.5767             | 0.5144            
0.40       | constant     | 0.1939             | 0.5144            
0.40       | 1/n          | 0.2518             | 0.4869            
0.40       | 1/sqrt(n)    | 0.5685             | 0.5708            
============================================================