# _Reinforcement Learning 101: AI plays Pokémon!_

Like many 90's kids I grew up playing Pokémon. I spent hundreds of hours on these games. Without easy access to guides some parts took serious time to figure out.
Now I'm a professional Data Scientist with an interest in Reinforcement Learning.
What interests me about Reinforcement Learning is that it's one of the few Machine Learning techniques that stays fairly close to how a human would approach the problem.

In this blog I will explain the basics of Reinforcement Learning and how each aspect can be applied to the Pokémon games.


## What is Reinforcement Learning?

Reinforcement Learning is a type of Machine Learning where an agent learns to make decisions by performing certain actions in an environment.
In short with the key terms in bold:

The **agent** uses a machine learning model that we aim to train.
The input for this model is the current **state** of the **environment**.
The output is the **action** it takes based on the input.
The **action** is chose gets executed in the **environment** resulting in a new **state** and a **reward**.
The **agent** uses the state-action-reward combination as training data and updates the model using a predefined **policy**.

All this is executed in a loop, each iteration feeding the new state to the updated agent, resulting in slowly improving actions.


### Environment 
This is the Pokémon game itself, including the world, battles, menus and all interactions. The environment has a variety of tasks within the training loop.

- Managing the emulator running the game. This includes loading, saving and resetting the game as needed.
- Executing actions on that emulator.
- Producing the state in a format that is ready for the model to interpret.
- Calculating the reward for a given action. This is done by reading the memory of the emulator for 
important in-game values like levels, world exploration and any other values that indicate game progression.

**Multiple of these environments** can be run in parallel. This not only speeds up the training significantly, but also avoids ending up in a local minimum.
Initially the agent moves randomly. With a single environment there is decent chance that the agent does not find the next big reward before the iteration is over.
When running 24 environments in parallel, the chance they all get stuck is a lot smaller. I chose 24 environments because that's the maximum I can fit in my RAM.


### Agent: The AI trainer
The agent is the player. The agent holds the model used to determine the action to take. 
The model is initialized randomly, so it starts with no knowledge and learns by trial-and-error.
Early on the actions will be random, but with each iteration of learning it gets less and less random.

There are **many different algorithms** the agent can use to update the model. Some don't even require a model at all.
Explaining the difference between these could fill a blog on its own.

To choose one for this project I compared the more popular algorithms. These are Deep Q-Networks (DQN), Proximal Policy Optimisation (PPO) and Advantage Actor Critic (A2C).
I found that: 

- **DQN** approximates a state-value function. This algorithm works best with discrete state spaces. 
Tic-tac-toe has such a state-space: {empty, X or O} per cell. A screenshot as used in this project is not discreet.
DQN is also off-policy, meaning it does not necessarily need a policy. It just learns from all the states it has seen so far to optimize the reward. This does come with some overhead, which I noticed in RAM-usage.
- **PPO** can be used for a wide variety of tasks, both in discrete and continuous state spaces. However, it will require more samples to train to the same level as other algorithms.
- **A2C** trains multiple neural nets, the actor and the critic, which comes with overhead in the learning phase.

As DQN doesn't handle continuous state-spaces well, it was easily disqualified.
PPO vs. A2C was close, but tests showed that PPO learned a lot faster. PPO required more iteration to get to the same point in the game, but A2C was slower overall, spending a lot more time learning instead of gathering more data.


### Policy: Improving strategies
The model is updated using a policy. Which policy you can or should use depends mostly on the complexity of your environment and the way you're going to feed the state to the model.

For example: For a model learning to play tic-tac-toe using a neural net would be overkill. The input is 3 by 3 and there are a very limited number of possible states to learn the 'correct' action for.
For Pokémon however, there are hundreds of variables plus the screen output you could use as input.
I chose to use just a screenshot as input for the model, making a policy that uses a Convolutional Neural Net the logical choice.


### State: Snapshot of the adventure
A screenshot of the current state of the game. This is fed to the model to determine the best action to take.

A state can also just be a list of variables. When using RL for something like stock trading, the input would be the stock prices, history, relevant news, etc.
But for this problem a screenshot is easy to obtain from the emulator and has all the info the model should need.


### Action: Decisive moves
The action the agent takes to influence the state of the environment in some way.
The action-space is the collection of actions the agent can do. In this case most of the buttons: ←↑→↓AB.
I chose to leave out the START and SELECT button, since you don't need them until halfway the game.


### Reward: The Heart of the Training Process
The reward is what it's all about. It is the only feedback the agent gets to update its model.
The Agent will train the model to maximize the reward, without knowing what it is made up of.

The reward can include anything you can get from the emulator. The most important parts of the reward were:
- Levels: The total level of your Pokémon. This makes sure that the agent keeps defeating wild Pokémon. It does run the risk of getting stuck in the first grass it sees, training its Pokémon to level 100.
- Opponent damage: Damage dealt to opponent Pokémon. This helped the agent understand how to defeat wild Pokémon at all. It will get a reward for using damaging moves, and no reward for using the moves it should avoid.
- Explore: A small reward per unique square visited in the world. This motivates the agent to explore the world as much as possible
- Negative steps: A tiny negative reward per action executed. This ensures that the agent will keep trying to find a next reward as it punished actions that don't result in any reward.

There were many more statistics factored into the reward.
Anything that you can read from the emulator memory can be used in the reward function.

The problem here is that Pokémon games don't have just one goal. Of course there is the slogan 'Gotta catch 'em all', but that is a goal few people achieve.
Another goal is beating all the gyms, important fights that get progressively harder throughout the game.
To achieve any of these goals, there are smaller tasks to complete. Pick up your first Pokémon, make your way to the next town, win battles etc. 
The reward function would be slightly different for each of these goals.
Crafting the perfect reward function is a big challenge, if not impossible, for a game like Pokémon.

## Conclusion: Gotta train 'em All
Combining Reinforcement Learning and Pokémon presents a fascinating mix of nostalgia and modern technology.
By applying the core concepts of Reinforcement Learning, we can have it play Pokémon relatively easily.

It took quite some training, but the agent made it to the first town. With more training capacity I have no doubt it could finish the game. And then what's next?
Compared to Atari, Pokémon is a relatively complex game, but there are way more complex games out there. I think it is only a matter of time before AI can play them all.
