# What is tested?

The full hypothesis is, when training a policy to control different bodies with different observation space,
 if we align the observation dimensions (keep positions of similar things constant in different space), it will perform better than 
 misalign the observation dimensions (random positions for observation).

# What are Treatment and Control groups?

In this experiment, we are testing:
Suppose there are one hopper and one ant, 
the treatment is align the obs carefully, the control is align the obs randomly.
Dimension of robots:
8 + 2x<joint> + <feet>
Ant has 8 joints and 4 feet, so 28 dimensions of observations.
Hopper has 3 joints and 1 feet, so 15 dimensions of observations.

Treatment:
My aligned way, training two bodies.

Control:
Other ways, training two bodies.

Will the treatment need less training steps? or can the treatment perform better using same training budge.

# What are the results? (in short)

No obvious difference when training two bodies with obs in different orders.

# Other Thoughts?

other thoughts from the Jan 5 meeting:

The CORE of a M.S. thesis is to prove there IS a clear contribution (no matter how small it is), and show experiments thoroghly investigating this question. 

Other parts could help but not the core, such as the story of where the question came from, the future questions triggered by this work, potential errors that might exist in other papers if ignore this work. 
e.g. we first hypothesized something else and the results came back negative, and we wanted to know why negative, and it lead us to this hypothesis. 
e.g. in this work we investigate the alignment problem in low-dimension observation, if this seems trivial, how should we align the observation for more complex robots, like voxcraft robots.
e.g. when people construct their baseline, if ignore this work, they might have a handicapped baseline.


Possible variables:
1. different bodies
2. different RL algorithms
3. different body type combinations
4. different random seeds
5. different ...