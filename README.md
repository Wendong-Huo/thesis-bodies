# This repo was forked from RL Baselines3 Zoo

# Change I'm going to make

1. Add PyBullet body databases;

    * toy_v5 added.

2. Test one Walker2D variation;
    
    * trained very slow. 2M steps can't make the bot to walk. Please check why. (test vanilla body dimension.)
    
    * the optimized RL is fragile. We need to keep all scaffolds to insure the speed of training.

    * Now building the pybullet_walker2d dataset. The first vanilla 2D walker is walking. tb/PPO_59 is vanilla.  tb/PPO_60 is only give +1 bonus for first 100 steps. 

    * We can't ignore live bonus and height and pitch, otherwise it will take very long to train.

    * Trying out different possible bodies. seems fly-away only happens when torso is too small and motor[0] is too big. other power_coef seems not causing fly-away. I don't know why.

    * Sweep 216 power coeff, see their learnability. I can sort them from easiest to hardest. and form different dataset.

    * I set power coeff proportional to the volumes of two connected parts, which is intuitive.

    * I generated 100 random bodies with variation=1/2 in dataset/walker2d_v6, and sent them to vacc to train individually. I need to make a program to visualize those 100 bodies and evaluate them and show the total reward and total distance.

    * make a training env try to get fly-away bugs. (oh, no, it's no the fly-away bug! it's the bug of shuffle in dataset... I forgot to shuffle params as well. fixed.)

    * trained 100 bodies with 3 seeds. sorted them and plot. now need to cross_test them and make a heat matrix. (random seed still have some problem, the model evaluated during training can't be reproduced on laptop.)

    * cross_test cross tests 100 bodies, producing a matrix tells by default, how is the generalization. (2 steps: measure and collect data on VACC (using 100 tasks), to cross_test.pickle, produce figure using plt.matshow on laptop. )

    * observation: seems training on bad body and testing on good body is slightly likely to get better result than reverse.
    

3. Modify PPO, concatenate body info to observation;

    * Adding 18-dim static param to observation, hope this won't hurt much. Train and test on the same body, same 3x100 carried out on VACC.

    * Observation space matters paper says adding body_x to observation can hurt learning process, I think it make sense because body_x is getting larger and larger, and it doesn't get normalized. So when the agent get some place far from origin, the input will becomes very large, which is bad for neural network. But in my experiment, it doesn't hurt No.9 much. (I only did one run. and I mailed them asking for source.)

4. Test one Walker2D variation with body info;

    * I divided 100 bodies into 10 groups. The best bodies are in the first group (group 0). The worst ones in group 9.

    * Train policy on a group of 10 bodies, with body info. (obs.space = original_obs.space + 18 params about body). Trained with three seed. 3xg10. so we will have 30 policies. (This is g10)

    * Test these policies in 100 bodies and produce heat matrix that can compare to the one in section 1.

    * train on a group of 20 bodies, (g20 train on sorted group, 20u train on unsorted group).
    
    * I am going to clean up the code and make it more reuseable. In Cleaned_version

5. Test

    * preliminary test results: in 900 tests, w/o bodyinfo 28 successes, w/ bodyinfo 38 successes.

    * I only train them for 2M steps. I'll increase training time, see if there will be more successes.
    
    * I should take 80% bodies to train, and 20% bodies to test, the data will be nicer.

    * I increased the replay buffer size according to the number of env.

    * two experiments started.

    * walker2d_20_10-v0 started on short
    
    * walker2d_30_20-v0 started on bluemoon