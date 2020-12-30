from stable_baselines3 import PPO

filename = "exp5/model-ant-400-500-600-s0.zip"

model = PPO.load(filename)

print(model)