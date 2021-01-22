from stable_baselines3 import PPO
from common import common
args = common.args
args.model_filename = "output_data/tmp/model-900-901-902-903-904-905-906-907-pns-pns_init-sd0.zip"
model = PPO.load(args.model_filename)
print(model)

for i in range(8):
    print(f"weight[{i}]")
    weights = model.policy.features_extractor.pns[i].weight.detach().numpy()
    # weights = .tolist()
    for j in range(weights.shape[0]):
        for k in range(weights.shape[1]):
            print(f"{weights[j,k]:.02f}, ", end="")
        print("")

    print("===========")