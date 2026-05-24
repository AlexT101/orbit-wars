# Orbit Wars | OW-Proto

## **Author notes:**  
*"This is the final version of Proto for now. The score peaked around **1080**, which was **Top 95**, stabilized between **1020-1050**, which was around **Top 110-130**. I feel I contributed a good bit to this competition, providing quality code for others to learn and take inspiration from. I will continue competing privately, and likely update this notebook in the future. Good luck to everyone competing and thanks for checking out my work!"* 😄

## Proto scoring formula
```python
score = (100 - dist) + (15 * t.production) + (10 * enemy_bonus) - (0.7 * total_ships) - (2 * eta)
```

Definitions:
- `dist`: distance from home planet to target planet.
- `t`: target planet.
- `enemy_bonus`: extra value when target planet is owned by opponents.
- `total_ships`: total ships needed for capture, including expected production if target is owned.
- `eta`: estimated fleet arrival time.

## Proto-V15 — Final LB Score: Peaked at 1080, stabilized around 1020-1050
Notebook version: **19, 21**

**If you want a more detailed version history, check out notebook versions 19 and under.**

**Final main features:**
- Can plan moving planet trajectories to calculate collision angle.
- Dynamic cooperative attacks when one planet cannot capture alone.
- Uses custom optimal target scoring formula.
- Never misses target planets.
- Avoids sending fleets into the sun, also avoids comets completely.
- Sophisticated defense system: calculates at exactly what tick a planet will be vulnerable. Reinforcements are sent and ensured to arrive before the enemy fleets arrive. A planet that's under attack will retain from sending more ships than they can afford, so they won't be left vulnerable.
- Friendly fleet trajectories are kept in mind. Home planets can calculate whether the fleets targeting an enemy planet are insufficient, and send the extra necessary ships.
- Planets are limited to sending max 1 fleet per tick.

**Main issues:**  
- Reinforcement system currently sees all planets as equals, but in reality a planet with production 5 is way more valuable than 1 or 2. Valuable planets are not being prioritized for reinforcements.
- Reinforcements from a planet get comepletely dropped if they won't be able to make it in time before the enemy fleet arrives. Problem with this is that the fleet speed is calculated using the requested ships amount, meaning that the *optimal* amount of resources won't arrive in time, however if planet is valuable (high production), then we could calculate the minimum amount of ships to send in a fleet that will arrive faster than the attacker's fleet, so we could still save the planet.
- Does not account for accidental collisions with other planets in the fleet path.
- A lot more, but that's up to you to fix. 😉
