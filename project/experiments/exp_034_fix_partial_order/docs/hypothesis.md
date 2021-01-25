# What is tested?

Previously in exp_033, cnspns is worse than expected.

In this experiment, I am going to fix the first body's sensor motor pns to be identity matrix, and keep the first 8 numbers of other sensor pns to be identity matrix.
So the prior knowledge "the general information has the same order" can be utilized.

the flag --cnspns_fix_general_info, --cnspns_start_with_identity, --cnspns_fix_pns_bodies are the new switches.


# What are Treatment and Control groups?

Control:
Train on one, with no cnspns

Treatment:
1. Train on one, start with identity matrix (pns), and fix pns, which should equivalent to the control.
2. Train on one, start with identity matrix (pns), and fix general info, should be slightly better than the control, since the pns module can be adjusted.
3. Train on one, start with identity matrix (pns), and fix general info, should be slightly better than the control, enable relu, since the pns module can be adjusted and there is additional non-linearity.

# What are the results? (in short)


# Other Thoughts?

Note: ent_ceof is set to 0, which agree with the original paper, but would it help Walker2D?