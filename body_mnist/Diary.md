* made a mistake during training. re-train 'with-bodyinfo' for all 3 cases.
  need to re-test these parts after done.

* make python file that can search for WORKABLE bodies.

* experiment "exp_3b" (experiment with three body classes)

  1. `generate_random_bodies.py`: create 500 random bodies;
  2. `train_one.py`: train 500 policies, test on themselves;
  3. manually pick 3 bodies as templates;
  4. change `body-templates/ant.yaml`, add 3 templates (for 3 classes);
  5. `generate_bodies_from_classes`: create 10 variations for each class;
  6. `start_exp.py`: randomly build 20 combinations (each sample uses 80% bodies for training and 20% bodies for testing), and train these 20 combinations;
  7. `test_simple.py`: test trained policies on target test bodies;


* Design:

  1. `train.py`: --exp, --seed, --train-bodies, --test-bodies (eval) --with-bodyinfo;
  2. `start_training.py`: --exp; sbatch all `train.py`.
  3. `test.py`: --exp, --seed, --train-with-bodyinfo, --test-as-bodyinfo, --train-bodies, --test-bodies, --render;
  4. `start_testing.py`: --exp; sbatch all `test.py`.
  5. `submit.sh`: submit an arbitrary job.

* exp_3b failed. Fall back to exp_2b (2 classes, 5 bodies in each class).
  "Don't jump to far, move one step at a time."
  

* exp_add_diff_help/tb/model-ant-0/PPO_1 => train: 0, test: 0
