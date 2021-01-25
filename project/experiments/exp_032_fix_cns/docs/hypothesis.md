# What is tested?

Now we relax the constraints: we allow a body come with a PNS system (one square linear layer with weights and bias).

Step 1. Train on 399. without any constraints on PNS.
Step 2. Train on 499, with CNS fixed.
Step 3. Train on 599, with CNS fixed.
Step 4. Train on 699, with CNS fixed.
Now we will have a policy that can work on V4.



# What are Treatment and Control groups?


# What are the results? (in short)

channel size of 32,32 is better than smaller such as 8,8 or 4,4. 
channel size of 128, 256 is even better, but that will enlarge the PNS nets and increase the time to train PNS.
I chose sensor=32, motor=8.

For small channel, s=8,m=4, 399 not working, one of 499 not working but Ants is very much fine.
Test on small channel again with adjustment in vec_normal and ppo clip_range, we have the very similar result above. (wired)

==
If we first train a CNS using 399, and use that to train 399, 499, 599, 699 together, it doesn't work. (indicate the training for 399 made a bad initialization?)

==
cnspns architecture makes training on one body harder.



# Other Thoughts?

