# Dec 17, 2020

* Body 0, 1 are different. Body 2 is similar to 0. Training on 0,1 without bodyinfo testing on 2 performs better than training on 0,1 with bodyinfo and testing on 2 with the knowledge that 2 is similar to 0.
Is this because the total training step is too small?

* exp_v4, train for 8e6 steps.

* exp_test_speed, train on 3 cpus per experiment, see if it will be faster. (No, no significant speedup.)

* exp_v5, record eval of 2 during training.