# What is tested?
To test: 
    Adding framestack help standard RL trained on multi-body perform better on multi-body?

# What are Treatment and Control groups?
Control: 
    No stacking.
Treatment: 
    Stacking 4 frames for observation (30,)->(120,)
    Stacking 8 frames for observation (30,)->(240,)
    Stacking 16 frames for observation (30,)->(480,)

# What are the results? (in short)

similar to train on one test on one.

The best Walker2D stand still.

in halfcheetah stackframe4 is better than 0

in ant and hopper, 0 is best.

# Other Thoughts?

stack-frames is an idea introduced in DeepMind’s two papers (from NIPS workshop 2013 and NATURE 2015) when they played Atari.

Also they introduced MaxAndSkip idea:
https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html#stable_baselines3.common.atari_wrappers.MaxAndSkipEnv

But this is for image input.

We should be able to use 4 frames, not consecutive ones but skip several steps. It should work even better.

A good blog article: 
https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/

From DQN 2015 paper:
Because running the emulator forward for one step requires much less computation than having the agent select an action, this technique allows the agent
to play roughly `k` times more games without significantly increasing the runtime.
We use `k = 4` for all games.

We only need skipping when 16 frames is better than 4 frames.
