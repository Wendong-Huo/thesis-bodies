import gym
import pybullet_envs
import numpy as np


sysprint = print
g_line = 0
def print(*args):
    global g_line
    CSI="\x1B["
    sysprint(CSI+"31;40m\n"+f"{g_line}> ", end="")
    sysprint(*args)
    sysprint(CSI + "0m")
    g_line+=1
def printt(*args):
    global g_line
    CSI="\x1B["
    sysprint(CSI+"32;40m\n"+f"{g_line}> ", end="")
    sysprint(*args)
    sysprint(CSI + "0m")
    g_line+=1


if __name__ == "__main__":
    
    env = gym.make("Walker2DBulletEnv-v0")
    env.seed(0)
    env.reset()
    env.action_space.seed(0)
    printt("joint position")
    print(env.env.robot.jdict["thigh_joint"].get_state())
    
    a = env.action_space.sample()
    total = 0
    printt("step and joint position")
    for i in range(5000):
        ret = env.step(a)
        total += env.env.robot.jdict["thigh_joint"].get_state()[1]
        # print(np.sum(ret[0]), ret[1], ret[2])
        # print(env.env.robot.jdict["thigh_joint"].get_state()[1])
        if ret[1]:
            env.reset()
    print(total)