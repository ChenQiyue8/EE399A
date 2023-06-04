In my experimental endeavors, I opted for a unique combination of hyperparameters:

n_episodes = 1000000 

epsilon = 0.05 

gamma = 0.875 

rewards = [-10, -1, 2, 1000000] # Death, ineffcient move, apple, win (occupy all blocks)

This setup contrasts with the original settings where epsilon was 0.3 and gamma 0.9, leading to a winning ratio of around 1 in 20 games. The modified parameters turned out to be significantly more effective.

The epsilon parameter stands for the exploration factor, demonstrating the agent's inclination towards investigating new and unknown environments. A higher epsilon indicates a stronger predisposition of the agent (the snake, in our case) to spend more time probing unexplored territories. However, it's essential to find an equilibrium between exploration and exploitation to yield optimal results.

A lower epsilon value, such as 0.05 in this scenario, means that the agent leans more towards exploiting known information rather than exploring unknown territories. This change resulted in better performance because the agent relied more heavily on the knowledge it had already gained, making decisions based on proven strategies rather than taking risks on unknown routes. As a result, it spent more time reinforcing successful behaviors, leading to a higher success rate.

The gamma value reflects the balance between immediate and future rewards, which becomes relevant when the environment doesn't promptly respond to the agent's actions. Sometimes, a series of actions is required before determining the effectiveness of a particular approach. With the updated parameters, my snake demonstrated a strikingly improved performance, securing victories in 33 out of 34 games.

I also found that when death reward is very negative, or when ineeficient move reward is not low enough, the snake tend to circle around and keep doing infinite loops. That is due to poor reward function causes the agent to be to scared of death or not being penalized by ineeficient moves enough, causing it to circle around. 

<p>
  <img src='https://github.com/qchen4/EE399A/blob/3307d39df3328e15102a46663d7c554d176725f4/HW7/Screenshot%202023-06-02%20at%2011.41.22%20AM.png'>
</p>

The image is the finial result of my hyperparameters, which leads me to win almost every game. 
