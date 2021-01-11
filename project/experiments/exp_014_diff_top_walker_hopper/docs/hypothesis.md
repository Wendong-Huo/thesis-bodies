# What is tested?

Now we have different topologies.
If we have a policy that can control both Walker2D and Hopper, how should we align the observation.

# What are Treatment and Control groups?

Treatment:
align by taking observation of [joints] and [feetcontact] in an arbitrary order and directly padding zero to the end.
e.g.
observation=[general|joints|feetcontact]
obs(Walker2D) = [0,1,2,3,4,5,6,7|08,09,10,11,12,13,14,15,16,17,18,19|20,21]
obs(Hopper)   = [0,1,2,3,4,5,6,7|10,11,12,13,08,09|14|--,--,--,--,--,--,--] (Case 1)
obs(Hopper)   = [0,1,2,3,4,5,6,7|12,13,08,09,10,11|14|--,--,--,--,--,--,--] (Case 2)

Control:
taking observation of [joints] and [feetcontact] in an arbitrary order, but align sections.
e.g.
obs(Walker2D) = [0,1,2,3,4,5,6,7|08,09,10,11,12,13,14,15,16,17,18,19|20,21]
obs(Hopper)   = [0,1,2,3,4,5,6,7|10,11,12,13,08,09,--,--,--,--,--,--|14,--] (Case 3)
obs(Hopper)   = [0,1,2,3,4,5,6,7|12,13,08,09,10,11,--,--,--,--,--,--|14,--] (Case 4)

align sections, and align joints by align similar positions (taking topology into account. joints and feet, but there's only one feet here)
e.g.
obs(Walker2D) = [0,1,2,3,4,5,6,7|08,09,10,11,12,13,14,15,16,17,18,19|20,21]
obs(Hopper)   = [0,1,2,3,4,5,6,7|08,09,10,11,12,13,--,--,--,--,--,--|14,--] (Case 5)
obs(Hopper)   = [0,1,2,3,4,5,6,7|--,--,--,--,--,--,08,09,10,11,12,13|14,--] (Case 6)

# What are the results? (in short)

Alignment seems doesn't matter in this experiment.

# Other Thoughts?

Add arms to walker2d?