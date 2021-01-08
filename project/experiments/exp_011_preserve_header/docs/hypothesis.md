# What is tested?

Suppose there are 16 body variations, and they are the same type, e.g. walker2d.
The control is training a policy on them.
The treatment is training a policy on them after shuffling the order of observations for all of them.

We do this for 10 runs, in 4 types of bodies.

Will there be any difference in performance?

# What are Treatment and Control groups?

If we only allow joints part and feet contact part to be randomized, what will the learning curve looks like in 16 same-topology bodies.
Comparing to previous oracle and aligned.

# What are the results? (in short)

the result of [joints_only] and [joints_feetcontact] is [-ph-pfc] and [-ph], respectively.

better than [general_joints_feetcontact], worse than [aligned], as expected.

see: plots/exp_all_values.png

# Other Thoughts?

take a look at the nodes that is running my jobs
```
squeue -u sliu1 -h | awk '{print $8}' | xargs -I {} scontrol show node {} | grep -P 'NodeName'
```