Hey everyone! This is my first time doing a code write up and I just wanted to share the core logic behind my agent that peaked at **1103** in the Orbit Wars challenge.

My bot's philosophy is to be **aggressively adaptive**. This approach can be described as a greedy planning loop which continually assesses possible moves in the current turn of play to identify the one(s) most likely to yield the greatest short term and intermediate gains. Now, I didn't intend on doing super deep, complex search trees because, honestly, the game is fast paced, and sometimes simple, robust heuristics beat over-optimization.

## Key Components of the Agent

### 1 (`WorldModel`)

so this basically takes the raw game observations and creates a easier representation of the universe, which includes:

-   **Planet Ownership & Resources:** Who owns what, how many ships they have, and their production rates.
-   **Fleet Tracking:** Where are all the fleets, and when will they arrive? (which is crucial for predicting future states)
-   **Game State Metrics:** Is it early game? Late game? Are we ahead or behind? (used for aggression level)

### 2 Precise Orbital Mechanics & Interception

Somethign I completely missed during my first one is that **planets move**! You can't just aim at their current position. My bot uses `estimate_arrival` and `aim_with_prediction` to calculate the exact trajectory needed to intercept a moving target, which includes:

-   **Fleet Speed Calculation:** Since not all ships travel at the same speed (eg.larger fleets are faster)
-   **Sun Avoidance:** We also HAVE to check if our path **crosses** the sun. If it does, we find a safe alternative or just don't send the fleet (dont lose your ship to the sun)

### 3 Dynamic Target Valuation

Not all planets are equal and we can value them with the following factors:

-   **Production Potential:** Planets that produce more ships are generally more valuable over time.
-   **Strategic Location:** Proximity to our planets or enemy planets matters
-   **Game Phase:** Early game - neutral planets are gold. Late game - eliminating an enemy's last planet is the goal
-   
### 4. Mission-Oriented Action Planning

Instead of just "attack," my bot thinks in terms of "missions":

-   **Capture:**  Take over neutral or enemy planets to expand our empire
-   **Reinforce:** Protecting existing planets from enemy threats or predicted falls. That is to say, sometimes the offense is a good defense
-   **Snipe:** Capture a neutral planet right after an enemy fleet hits it where you essentially steal them while they're low hp
-   **Swarm:** Coordinating multiple fleets to hit a single high value target simultaneously

### 5 Greedy Execution with Resource Management

Each turn, the bot generates a list of all possible missions, scores them, and then executes the highest-scoring ones. Sure, you can argue that it's a greedy approach, but it semiworks because we're constantly re-evaluating. 

### Code Structure & Style:

The code is written in Python, and I tried my best  to keep it modular. You'll find constants at the top for easy tuning. You might notice some unconventional spacing or comments (I only wanted to get the logic down).

Overall, I hope this overview helps you understand how my thought process behind a good Orbit bot works. Feel free to fork it and build your own strategies. Good luck!
