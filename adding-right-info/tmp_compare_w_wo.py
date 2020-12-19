f1 = "model-ant-0-1.zip"
f2 = "model-ant-0-1-with-bodyinfo.zip"
from stable_baselines3 import PPO


def show(fname):
    print(f"Showing {fname}")
    model = PPO.load(fname)
    p = model.policy.mlp_extractor.policy_net.parameters()
    p = list(p)
    print(p[0])

show(f1)
show(f2)