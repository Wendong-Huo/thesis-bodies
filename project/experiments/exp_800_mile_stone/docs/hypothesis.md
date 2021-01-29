# Common Methods

We use PyBullet to simulate physics.

Learnability is defined to be the expected episodic reward the robot gets in a testing episode (with a certain testing body) after a certain amount of training.

L(b, t)

where L is the learnability, b is the body, t is the timesteps of training.

# Baseline

800.1.methods
To make a baseline, we train a policy on each of Walker2D, HalfCheetah, Ant, and Hopper, using RL (we used PPO implemented in stable-baselines3, we will keep the program and hyperparameters the same throughout this thesis). We repeat this for 10 different seeds.

800.1.results
The learning curve of Walker2D, HalfCheetah, Ant, and Hopper. and report the mean learnability at step 2e6.


# Exp 1. Alignments in parametrically different robots

##### What is tested?

Suppose we have a set of n parametrically different robots, and we want one policy to control them.

Hypothesis 1:
Does misaligning the observation and action spaces affect the learnability?

##### Methods

801.1.methods
Based on Walker2D, HalfCheetah, Ant, and Hopper, we randomly vary the length and thickness of the limbs (including the torso), 
and we get 100 robots of each type.

801.1.results
All the 400 robots generated.


801.2.methods
For each type, 
we train the 100 robots one by one; repeat for 3 seeds;
We are going to pick 16 of those with highest learnability (measured at 2e6 steps);
This is to make sure we have valid test-beds (robots) in our experiments.
So we have 16 robots of each type.
### This is not used. We actually selected body by training using stack frame 4 and by shortest training time to pass learnability of 1500.

801.2.results
801.2.1 All 16*4 robots selected (add a mark to 801.1.results).
801.2.2 The distribution of learnability per type. (histogram or density plot with y-axis represents the learnability) (add a threshold to show our selection) (add another line to indicate the baseline as a horizontal green dashed line)


801.3.methods
For each type,
we randomly pick two robots, and train a RL policy on them, with the observation and action space aligned, we measure the learnability at 2e6 steps.
for the same two robots, we train another RL policy on them, with the observation and action space randomized, we measure the learnability at 2e6 steps.
We repeat the two experiments 10 times, so we have a comparison between aligned and randomized when training on 2 robots.
We then repeat the same procedures on 4 robots, 8 robots, 16 robots, and have comparisons for those conditions.

801.3.results
801.3.1 A multi-plot grid, with different numbers of robots that are trained as rows, and with different types as columns.
For each plot, there are two learning curves. x-axis is t, y-axis is the learnability. (color and label: blue for "aligned", red for "randomized") (add the baseline)


# Exp 2. Alignments in topologically different robots (without obvious correspondence)

##### What is tested?

Suppose we have a set of n topologically different robots with no obvious correspondence, and we want one policy to control them.
People often use an arbitrary alignment (which often is the default orders come with the robots).

Hypothesis 2:
Do different alignments affect learnability?

##### Methods

802.1.methods
We choose 4 robots from PyBullet: Walker2D, HalfCheetah, Ant, and Hopper. (NOT the variations generated in 801.1)
Randomly generate 2 alignments, #1 and #2.
Train a policy on these four robots with the alignment #1. Repeat this training 100 times with different seeds.
Train a policy on these four robots with the alignment #2. Repeat this training 100 times with the same set of seeds we used in the line above.

802.1.results
802.1.1 learning curve comparison. we compare the learning curve of #1 and #2. (add the baseline)
802.1.2 a bar chart of the learnability at step 2e6.
802.1.3 distribution of learnability at step 2e6. (use histogram or density plot, and make y-axis to be the learnability.) Also report the p-value at step 2e6.

802.2.methods
To test robots other than those in 802.1. we randomly generate 100 tree-like robots by adding random limbs to an existing limb for each robot.
We limit the limb number to be <= 6, and the depth of the tree to be <=4.

802.2.results
All 100 random tree-like robots generated.

802.3.methods
Similar to 801.2.
we train the 100 robots one by one using RL (we used PPO implemented in stable-baselines3); repeat for 3 seeds;
We pick 8 of those with highest learnability (measured at 2e6 steps);

802.3.results
802.3.1 All 8 robots selected (add a mark to 802.2.results).
802.3.2 The distribution of learnability. (histogram or density plot with y-axis represents the learnability) (add a threshold to show our selection)

802.3.methods
Repeat 802.1.methods with 8 selected robots (802.3.results).

802.3.results
802.3.1 learning curve comparison
802.3.2 a bar chart of learnability at step 2e6.
802.3.3 a distribution of learnability at step 2e6. and report the p-value.


# Exp 3. Alignments in topologically different robots with obvious correspondence

##### What is tested?

We construct a set of 8 topologically different robots, but we made them so that there is obvious correspondence among them.
In this special case, we can guess that the good alignment is according to the functional meaning of each joints and limbs, we call the good alignment M0.

We define permutation distance d between an alignment Md and M0 to be the number of random swap to produce Md from M0.
So M2 means start with M0, randomly select a robot (there are 8 robots in the set), and randomly select two different positions, and swap the order number, and randomly select a robot again, and random select two different positions, and do swap.
For example, suppose alignment M0 is [1,2,3::1,2,3::1,2,3], M1 can be [1,2,3::2,1,3::1,2,3], and M2 can be [1,2,3::2,1,3::3,2,1].

Note: the alignment solution space is a high dimensional space, and using one number d to estimate the distance is not accurate when d is large. But for small d, it is good.

Hypothesis 3:
Near M0, 

##### Methods

803.1.methods
Based on Walker2D, we add two small limbs to different limbs to create new robots. By doing so, we create 8 topologically different robots.

803.1.results
8 Walker2Ds with small limbs.

803.2.methods
Train on 8 robots one by one for 10 times each.

803.2.results
The learning curve of 8 robots. (add a baseline of Walker2D)

803.3.methods
Train a policy on these 8 robots with alignment M0, repeat with different seeds 5 times.
Train a policy on these 8 robots with 21 random M2 alignments, repeat with different seeds 5 times.
Pick the best in 21 and the worst in 21 to show the difference.
Train a policy on these 8 robots with 21 random M4 alignments, repeat with different seeds 5 times.
Train a policy on these 8 robots with 21 random M8 alignments, repeat with different seeds 5 times.
Train a policy on these 8 robots with 21 random M16 alignments, repeat with different seeds 5 times.
Train a policy on these 8 robots with 21 random M32 alignments, repeat with different seeds 5 times.


803.3.results
803.3.1 A multi-plot grid of learning curves with cols for different robots and rows for different permutation distance. (to all plots add a baseline of Walker2D), each plot contains two curves (color and label: blue for "best in 20", red for "worst in 20")
803.3.2 A averaged learning curve on top, and a line plot on bottom, x-axis is the permutation distance, y-axis is the averaged learnability. Showing the gradient from M0 to M32.

Note: it is easy to generate M_{t+1} from M_t, however, the chance of getting M_t back from M_{t+1} is small. So searching for M0 is hard.

803.4.methods
Confirm the significance.
Pick the best and the worst from each set of 20 random alignments in 803.3. Train repeatedly with different seeds 10 times.

803.4.results
The averaged learning curve. and report the p-value at step 2e6.

