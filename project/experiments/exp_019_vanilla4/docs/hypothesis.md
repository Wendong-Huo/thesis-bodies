# What is tested?

Vanilla4 = Walker2D, HalfCheetah, Ant, Hopper from pybullet

Generate 500 random alignments for these four bodies.

Train a policy for each alignment.

Find the best alignment and the worst alignment.

Send the best and the worst back for 20 runs, to confirm there is difference in the alignment.

Combined Value = 2xA + B + C + 0.5xD (Because of the distribution.)

# What are Treatment and Control groups?

Treatment: good alignment

Control: bad alignment

# What are the results? (in short)

The 500 random alignments follow some distribution. (plots: Walker2D like a bi-model, Ant is more normal distribution.)

Results for 20 runs of the good and bad alignment:

significantly different? Nope

Might because the initialization of the policy network is too important, so only use 1 run to search for good alignment is not a good way, because when 
seed changes, the result might changes.
Solution: In random search, we need to use multiple runs with different seeds for 1 alignment.

# Other Thoughts?

Let's revisit random bodies. I found a bug in previous code! :)