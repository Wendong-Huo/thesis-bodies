# What is tested?
To test: 
    Adding framestack really helps single-body training? More seeds!

# What are Treatment and Control groups?
Control: 
    No stacking.
Treatment: 
    Stacking 4 frames for observation (30,)->(120,)
    Stacking 8 frames for observation (30,)->(240,)
    Stacking 16 frames for observation (30,)->(480,)

# What are the results? (in short)

still only better in body walker2d. (some of non-stack-frame trails only learn to stand.)

stack frame 4 is better than 8 and 16

in ant, 0 > 4 > 8 > 16, interesting.

# Other Thoughts?

Since we acheived 2164 for pybullet walker2d, do we want to try SAC+gSDE+stackframe4, to see if we can achieve the best score?