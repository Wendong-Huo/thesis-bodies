import time
import ray
import gym
import numpy as np
import pyrobotdesign_env

grammar_file = "grammar_apr30.dot"

# @ray.remote(num_cpus=1)
def sim(robo_id):
    np.random.seed(0)
    with open(f"../input_data/robo/{1000+robo_id}.txt", "r") as f:
        str_rule = f.readline()
    env = gym.make("RobotLocomotion-v0", grammar_file=grammar_file, rule_sequence=str_rule)
    print(f"Robo {robo_id}: action dim {env.action_dim}.")
    env.render()
    env.seed(0)
    env.action_space.seed(0)
    env.reset()
    rewards = 0
    for i in range(100):
        a = env.action_space.sample()
        obs, r, done, _ = env.step(a)
        rewards += r
        # time.sleep(0.01)
    print(rewards)

if __name__ == "__main__":

    # ray.init(local_mode=True)
    ids = []
    for i in range(10):
        # ids.append(sim.remote(i))
        sim(i)

    # ray.wait(ids)