# What is tested?
To test: 
    Adding framestack with skip-frame help standard RL trained on multi-body perform better on multi-body?

# What are Treatment and Control groups?
Control: 
    No stacking.
Treatment: 
    Stacking 4 frames for observation (30,)->(120,) with skip-frame
    Stacking 8 frames for observation (30,)->(240,) with skip-frame
    <!-- Stacking 16 frames for observation (30,)->(480,) -->

# What are the results? (in short)

No, skipping makes things worse, the agents never learned to do meaningful things in the treatment experiments.

# Other Thoughts?

Forget about skipping