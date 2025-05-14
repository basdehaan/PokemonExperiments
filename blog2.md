# _Reinforcement Learning 201: AI plays ALL the Pokémon!_

Reinforcement learning (RL) has proven to be a powerful tool in mastering complex environments, from board games to robotics. But what happens when you train an RL model across multiple environments that are functionally similar but visually distinct—like two different Pokémon games?

In this post, we’ll explore the challenges and potential advantages of multi-environment reinforcement learning. Specifically, we’ll look at what it means to train an agent that can play and generalize across two visually different Pokémon games that share core mechanics but vary in their graphical representation. In this case I'll be using Pokémon Red and Pokémon Gold.

For this post I will assume you are familiar with the basic concepts of reinforcement learning. If not, I have written an earlier blog here: [Reinforcement Learning 101: AI plays Pokémon](https://medium.com/ordina-data/reinforcement-learning-101-ai-plays-pok%C3%A9mon-e0626bd6beae)

## Core Challenges

Training a single model on multiple different games comes with its challenges. I will go over the most important ones and later discuss the strategies you can apply to mitigate these challenges.


**Visual Discrepancy**

  Even though both Pokémon games generally follow the same rules, visually they're very different. The most obvious difference is that one is black and white, and one has color. The Convolutional Neural Net (CNN) that is used by the agent will need some training before realising game states that look different, are actually very similar. Shown below is an example of such a situation during training.

  ![multi_env_screenshot.png](multi_env_screenshot.png)

**Alignment of Action Spaces**

  In the case of these specific Pokémon games, this is not an issue. Because they were originally played on the same device, the controls are the same. Only when adding the next game, more buttons become available. In later versions a touchpad was added in addition to the existing buttons.


**Training Stability and Sample Efficiency**

  Adding environment variability often makes training slower and more unstable. The agent must see more data to understand what is invariant across the games. This can dramatically increase the number of episodes required to converge on effective strategies.


**Credit Assignment and Exploration**

  Since visual cues differ, it may take longer for the agent to recognize the causal relationships between its actions and the resulting rewards. Exploration strategies tuned for one environment might also fail in the other, leading to uneven performance or stagnation in one game. All values used to calculate rewards are taken directly from the emulator memory have to be tailored to the game being played.

## Key Advantages

TODO: some text, not just a list

**Improved Generalization**

  When done right, training across similar but distinct games forces the agent to latch onto higher-level abstractions—like game rules or reward structures—instead of memorizing visuals. This leads to an agent that can generalize better, not just to these two games but potentially to future unseen ones with similar mechanics.


**Robustness to Noise and Style Variations**

  An agent trained on multiple visual styles is more robust to graphical glitches, modded environments, or even fan-made versions. This is especially useful in the Pokémon community, where ROM hacks and custom versions are common.


**Zero-Shot Transfer Potential**

  If trained well, such an agent could play a new Pokémon game with no further training—a milestone in zero-shot transfer learning.  This is the next step of this project. I will evaluate the models performance by having it play the next generation of Pokémon games without any further training.


**Foundation for Modular or Meta-Learning**

  Multi-environment training lays the groundwork for more advanced systems, such as meta-RL, where the agent learns how to learn. It also supports modular architectures where perception and decision-making modules can be swapped or specialized per environment.


## Strategies to Mitigate Challenges

TODO: more text here, link every strat to a challenge?

**Domain Randomization**: Introduce controlled randomness in visual elements during training (color shifts, resolution changes, HUD modifications) to help the model generalize.


**Feature-Based Learning**: Extract and use symbolic representations or game states (if available) rather than raw pixels to reduce reliance on visuals. In the case of these Pokémon games this is relatively easy because the emulator they run in is also written in Python.


**Curriculum Learning**: Start training in a single game and progressively introduce the second one, or vice versa, to ease the adaptation process. This way the CNN model used is trained well on one visual and only has to be modified to also fit the new visuals. All the logic that comes after is already in place.


**Shared Encoders with Environment Tags**: Use a shared neural encoder with a small conditioning signal that indicates which game is currently being played. Adding 5 pixels to the top of the input of which you color either the left or right half white can serve as a trigger to handle the rest of the input differently.

## Conclusion

Training an RL agent across two visually distinct but mechanically similar Pokémon games is both a technical challenge and an exciting opportunity. It pushes the boundaries of generalization, robustness, and adaptability in reinforcement learning.

By tackling the visual domain gap and building abstraction into our agents, we get closer to AI systems that don’t just memorize games—but truly understand how to play them.