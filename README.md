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

    * 

4. Test one Walker2D variation with body info;

5. Test ...