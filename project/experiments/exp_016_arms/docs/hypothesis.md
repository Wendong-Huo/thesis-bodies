# What is tested?

Now we have different topologies. 
If we have a policy that can control both WalkerArms and Walker2D, how should we align the observation.

# What are Treatment and Control groups?

Control
align by taking observation of [joints] and [feetcontact] in an arbitrary order and directly padding zero to the end.
e.g.
observation=[general|joints|feetcontact]
obs(WalkerArms) = [0,1,2,3,4,5,6,7|08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23|24,25]
obs(Walker2D)   = [0,1,2,3,4,5,6,7|08,09,10,11,12,13,14,15,16,17,18,19|20,21]
Case1(Walker2D) = [0,1,2,3,4,5,6,7|08,09,10,11,12,13,14,15,16,17,18,19|20,21,--,--,--,--]

Control:
taking observation of [joints] and [feetcontact] in an arbitrary order, but align sections.
e.g.
obs(WalkerArms) = [0,1,2,3,4,5,6,7|08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23|24,25]
Case2(Walker2D) = [0,1,2,3,4,5,6,7|08,09,10,11,12,13,14,15,16,17,18,19,--,--,--,--|20,21]

align sections, and align joints by align similar positions (taking topology into account. joints and feet, but there's only one feet here)
e.g.
obs(WalkerArms) = [0,1,2,3,4,5,6,7|08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23|24,25]
Case3(Walker2D) = [0,1,2,3,4,5,6,7|--,--,--,--,08,09,10,11,12,13,14,15,16,17,18,19|20,21]

# What are the results? (in short)

Misalign mutants is worse than meaningful aligned mutants. (Should be the main preliminary experiment.)

# Other Thoughts?

results in tensorboard_9xx
videos in exp_018 data folder
