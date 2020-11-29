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

    * Now the replay buffer size is too big. The learning is super slow. Shouldn't replay buffer size change according to the performance?

    * Apply a simple strategy: when reward<1000, use small buf size to speedup learning. when reward>1000, use large buf size to stable learned.
    
    * Start Experiment: walker2d_20_10-v0 on partition bluemoon (canceled); walker2d_10_10-v1 on partition short (finished).

    * Use `proxy_uvm.sh` start a proxy, and use Firefox to access Web inside UVM.

    * Start Experiment: walker2d_20_10-v1, walker2d_20_10-v2, walker2d_100_10-v0, walker2d_100_10-v1, walker2d_100_30-v0 on partition ib.

    * After look at the learning curve, give a reasonable max_step for training. maybe 3e6 (10_10 indicates that) or 5e6 (more body or high variation need more steps?).

    * Planning experiments for two plots, y-axis is the distance, x-axis is the following: 

        1. with variation = 10%, num-bodies = [10, 20, 50, 100]

        2. with num-bodies = 20, variation = [10%, 20%, 50%, 90%]

        * commands:

        ```bash
            python main.py --vacc -p ib --num-bodies=10 --body-variation-range=10 --seed-bodies=10 --no-single --n-timesteps=5e6
            python main.py --vacc -p ib --num-bodies=20 --body-variation-range=10 --seed-bodies=10 --no-single --n-timesteps=5e6
            python main.py --vacc -p ib --num-bodies=50 --body-variation-range=10 --seed-bodies=10 --no-single --n-timesteps=5e6
            python main.py --vacc -p ib --num-bodies=100 --body-variation-range=10 --seed-bodies=10 --no-single --n-timesteps=5e6
            python main.py --vacc -p ib --num-bodies=20 --body-variation-range=20 --seed-bodies=10 --no-single --n-timesteps=5e6
            python main.py --vacc -p ib --num-bodies=20 --body-variation-range=50 --seed-bodies=10 --no-single --n-timesteps=5e6
            python main.py --vacc -p ib --num-bodies=20 --body-variation-range=90 --seed-bodies=10 --no-single --n-timesteps=5e6
        ```

    * Oh, don't use partition `ib`, that stands for Infinity Bendwidth.
    
    * Not significant:

        * body: 10, variation: 10% => 33.7 +- 4.4 vs 31.7 +- 2.2 (body:i3_s0,i7_s0,i7_s1 is missing; nobody:i1_s0, i1_s1, i2_s0, i3_s1, i4_s1, i6_s0, i8_s1 is missing.)

        * body: 20, variation: 10% => 35.0 +- 4.2 vs 33.4 +- 4.2

        * body: 50, variation: 10% => 26.2 +- 5.1 vs 28.1 +- 6.0

        * body: 100, variation: 10% => 9.6 +- 10.2 vs 8.5 +- 9.3 (stop earlier at 4e6, not 5e6)

        * body: 20, variation: 20% => 33.0 +- 6.9 vs 28.8 +- 5.0

        * body: 20, variation: 50% => 7.6 +- 6.1 vs 8.8 +- 6.9

        * body: 20, variation: 90% => 1.5 +- 0.8 vs 1.6 +- 1.0

    * Plot is in folder `_report_utils`. At least we now know blindly adding body information doesn't help.

    * Korea animation paper uses [128,128,128] network. so let's try it. Start [20, 30%], [20, 40%], [20, 50%] experiment on partition short.

    * 3-layer network doesn't help.
    

6. Test again more carefully.

    * First, send 20% 20, with 30 replicates, with 5 unique seeds. (total tasks = 2 x 30 x 5 = 300.) `walker2d_20_20-v40`

    * Second, the rest 4500 tasks.

    * 20_20 is still not significant:
    
    ```
    (should have 600, but some of them stopped accidentally.)
    body group has 556 non-zero value.
    nobody group has 568 non-zero value.
    with body) mean: 23.78560829012514, std: 7.030475298774249
    without body) mean: 23.0331832302708, std: 7.904171451290839
    ```

7. Open up the neural network

    * input space with body information:

        * in total: 39

        * so called "more"
        
            * 0: relative height, usually negative, z - self.initial_z

            * 1-2: angle to target, always zero for Walker2D, sin(angle_to_target), cos(angle_to_target)

            * 3-5: speed of the "torso" (rotate back in yaw, which doesn't change anything),  vx, vy, vz, (vy==0 for Walker2D)

            * 6: Roll of body, always zero for Walker2D.

            * 7: Pitch of body, facing up or facing down.

        * 8-20: there are 6 joints in Walker2D
            
            * 8: the position of the first joint, scaled to -1, 1, with a little overshoot.
            
            * 9: change in position of the first joint. (however not in the same scale with position.)

            * 10-20: other joints.
        
        * 21-22: feet contact. 0 for no, 1 for yes. two feet.

        * 23-39: body params from our param file.


7. Argument

    * Changes in parameter and topology. People usually think topology is more important than parameter changes. But if we think about this https://sustainablefisheries-uw.org/wp-content/uploads/2018/06/ShingletonFigure-1.jpg, we will realize parameter is important.

    