# What is tested?

Can we create a body dataset,
containing 20 Walker2D (300s) variations, 20 HalfCheetah (400s) variations, 20 Ant (500s) variations, 20 Hopper (600s) variations,
80 bodies in total.

All the bodies can be quickly trained using PPO (with stack_frames=4).

# What are Treatment and Control groups?

# What are the results? (in short)
80 bodies in output_data/selected_bodies

# Other Thoughts?
I didn't finish evaluating bodies with stack_frames=1, so I used stack_frames=4.
Be careful don't use this dataset to test if stack_frames is better than non-stack_Frames, because it was selected by sorting.
