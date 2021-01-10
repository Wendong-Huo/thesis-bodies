# What is tested?

Still we have 16 bodies. (later we will test 2/4/8 bodies)

All bodies have the same topology, the dimensions of bodies slightly vary. (later we will test bodies with different topologies)

These cases are:

1. Aligned. [done]
2. Only randomize "General" part. [-general_only]
3. Only randomize "Joints" part. [done] old name [-ph-pfc] new name [-joints_only]
4. Only randomize "FeetContact" part. [-feetcontact_only]
5. Randomize "General" and "Joints" parts. [-general_joints]
6. Randomize "General" and "FeetContact" parts. [-general_feetcontact]
7. Randomize "Joints" and "FeetContact" parts. [done] old name [-ph] new name [-joints_feetcontact]

And another compare curve 0. Oracle. (Train and test on the same body) [done] 

# What are Treatment and Control groups?

Treatment:
Case 2,4,5,6

Control:
we have controled cases "Aligned"

# What are the results? (in short)

In short: N=5?

For data of treatment and control:
average over: 5 seeds x body numbers (2,4,8,16) = 10,20,40,80
total runs: seeds(5) x num body numbers(4) x num types(4) x num methods(8) = 640
e.g. seed=0, body number=16, type=hopper, method=aligned.

For oracle:
average over: 2 seeds x 16 bodies = 32
total runs: seeds(2) x num possible bodies(20) x num types(4) = 160
e.g. seed=0, body=19, type=hopper

If there is misalignment, in general, performance hurts.

This is getting more obvious when number of bodies gets large. This is more obvious in 16 bodies than in 2 bodies.

Misaligning some dimensions hurts more than others in general, but differs with morphologies.

# Other Thoughts?

Re-train oracle for 5e6 steps.
Re-train everything for 5e6 steps.
There's something strange happen at about 2e6!
