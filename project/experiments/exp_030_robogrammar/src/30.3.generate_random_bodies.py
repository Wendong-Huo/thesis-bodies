import time, math
import gym
import numpy as np
import pyrobotdesign_env
from common import seeds
with open("../input_data/robo_grammar/grammar_apr30.dot", "r") as f:
    lines = f.readlines()
idx = 0
for line in lines:
    if line.startswith("digraph"):
        print(f"{idx} {line.split(' ')[1]}")
        idx += 1
# 0 make_robot
# 1 append_body
# 2 make_body_with_legs
# 3 make_body_without_legs
# 4 append_limb_link
# 5 end_limb
# 6 end_tail
# 7 end_head
# 8 make_normal_limb_link
# 9 make_long_limb_link
# 10 make_fixed_body_joint
# 11 make_roll_body_joint
# 12 make_swing_body_joint
# 13 make_lift_body_joint
# 14 make_left_roll_limb_joint
# 15 make_right_roll_limb_joint
# 16 make_swing_limb_joint
# 17 make_acute_lift_limb_joint
# 18 make_obtuse_lift_limb_joint
# 19 make_backwards_lift_limb_joint
rule_set = [1,2,4,8,9,11,13,16,17]
# rule_set = list(range(20))
with seeds.temp_seed(0):
    rule_sequences = np.random.choice(rule_set,size=[1000000,30])

grammar_file = "grammar_apr30.dot"
body_id = 0
for i in range(1000000):
    str_rule = ','.join(str(x) for x in rule_sequences[i])
    str_rule = f'0,{str_rule}'
    try:
        env = gym.make("RobotLocomotion-v0", grammar_file=grammar_file, rule_sequence=str_rule)
        # env.render()
        np.random.seed(0)
        env.seed(0)
        env.action_space.seed(0)
        env.reset()
        # env.render()
        print(f"{env.action_dim} joints in Rule {i}")
        if env.action_dim>=1:
            rewards = 0
            for i in range(1000):
                action = env.action_space.sample()
                obs, r, done, _ = env.step(action)
                base_tf = np.zeros((4, 4), order = 'f')
                env.sim.get_link_transform(env.robot_index, 0, base_tf)
                base_pos = base_tf[0:3, 3]
                base_pos_x = base_pos[0]
                # print(f"{base_pos[0]:.02f}, {base_pos[1]:.02f}, {base_pos[2]:.02f},")
                if done or math.isnan(r):
                    break
            final_pos = base_pos_x
            print(f"random action final_pos: {final_pos}")
            if final_pos>0:
                print(f"{'-'*30}write to file {1100+body_id}.txt")
                with open(f"output_data/bodies/{1100+body_id}.txt", "w") as f:
                    f.write(str_rule)
                body_id+=1
                if body_id>=10:
                    break
    except RuntimeError:
        pass
