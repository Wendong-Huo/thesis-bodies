# Body MNIST for RL

## The simplest case

Suppose we have 2 sets of robots: A and B. A = {a_i}, B = {b_i}. Bodies in one set are very similar to each other (simple case).

We sample from A and B to form the training set Train = {a0, a1, a2, a3, b0, b1, b2, b3}.

We sample from A and B to form the testing set Test = {a4, b4}

Control (Standard RL):

We train a policy using Train.

We test the policy on Test.

Treatment (with right information):

We train a policy using Train with body information (A or B).

We test the policy on Test with body information.

We will show Treatment is better than Control.
