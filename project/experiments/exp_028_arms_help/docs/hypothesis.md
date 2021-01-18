# What is tested?

With those baselines evaluated in exp_026, now test do different randomized alignments perform differently?

Why we still test 9xx? Because for a random body set, the alignment solution space is vast, and good alignment is hard to find.
Thus we construct the experiment 9xx so that we know there is a best solution and where it is. 
A efficient way of searching for good alignment is beyond the scope of this thesis.

# What are Treatment and Control groups?

Control:
```
    # bodies with randomly added arms (and the alignment partially breaks)
    sbatch -J baseline9xx submit.sh python 1.train.py --seed=$seed --train_bodies=900,901,902,903,904,905,906,907 --test_bodies=900,901,902,903,904,905,906,907 --train_steps=1e7

    # randomly generated bodies together (and the alignment is an arbitrary one)
    sbatch -J baseline1xx submit.sh python 1.train.py --seed=$seed --train_bodies=100,101,102,103,104,105,106,107 --test_bodies=100,101,102,103,104,105,106,107 --train_steps=1e7
```

# What are the results? (in short)


# Other Thoughts?

It is hard to find a good alignment. Many bad alignments don't have a good score, there won't be a big difference.

Start from a good alignment, and do modification, then we might have some good alignments that have significant difference.

A good alignment is:

0: arm, 
1: arm_left, 
2: thigh, 
3: leg, 
4: foot,
5: thigh_left,
6: leg_left,
7: foot_left.

900: 0,1,2,3,4,5,6,7
901: 1,2,0,3,4,5,6,7
902: 1,2,3,0,4,5,6,7
903: 1,2,3,4,5,0,6,7
904: 1,2,3,4,5,6,0,7
905: 1,0,2,3,4,5,6,7
906: 2,0,1,3,4,5,6,7
907: 2,3,0,1,4,5,6,7
