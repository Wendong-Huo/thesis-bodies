# What is tested?

Suppose there are 16 body variations, and they are the same type, e.g. walker2d.
The control is training a policy on them.
The treatment is training a policy on them after shuffling the order of observations for all of them.

We do this for 10 runs, in 4 types of bodies.

Will there be any difference in performance?

# What are Treatment and Control groups?

Treatment:
16 walkers with different orders

Control:
16 walkers with the same order.

# What are the results? (in short)

The difference is clear!

# Other Thoughts?

Waiting to add "oracle" curve, which is still running.
