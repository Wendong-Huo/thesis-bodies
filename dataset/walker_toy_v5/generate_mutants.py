import string
import numpy as np
import pickle
import yaml

np.random.seed(0)

def read_template(filename):
    with open(filename, "r") as f:
        _content = f.read()
    return string.Template(_content)

def write_xml(filename, data, body_xml):
    _content = body_xml.safe_substitute(data)
    with open(filename, "w") as f:
        print(_content, file=f)

def write_py(filename, data, walker2d_py):
    _content = walker2d_py.safe_substitute(data)
    with open(filename, "w") as f:
        print(_content, file=f)

def write_yaml(filename, data):
    with open(filename, "w") as f:
        yaml.dump(data,f)

def random(mean, vary=0.5):
    low = mean*(1-vary)
    high = mean*(1+vary)
    r = np.random.random()
    r = (high-low) * r + low
    return r

body_xml = read_template(f"./body.template")
walker2d_py = read_template(f"./walker2d.template")

variation = 0.4 # range of random variation.
fixed_distance_to_ground = 0.1

# expected length and weight: mean
original_limb_length = np.array([0.3, 0.6, 0.6, 0.2])
original_limb_weight = np.array([0.06, 0.06, 0.06, 0.08])

file_list = []
param_list = []
power_coef_list = []
for mutant_id in range(5):
    limb_length = original_limb_length.copy()
    limb_weight = original_limb_weight.copy()
    for i in range(len(limb_length)):
        limb_length[i] = random(limb_length[i], vary=variation)
    for i in range(len(limb_weight)):
        limb_weight[i] = random(limb_weight[i], vary=variation)

    volumes = limb_length * 3.14 * limb_weight * limb_weight
    volumes_int = (volumes * 10000).astype(int)
    power_coef_list.append(volumes_int)

    data = {
        "z0": limb_length[0] + limb_length[1] + limb_length[2] + fixed_distance_to_ground,
        "z1": limb_length[1] + limb_length[2] + fixed_distance_to_ground,
        "z2": limb_length[2] + fixed_distance_to_ground,
        "z3": fixed_distance_to_ground,
        "x3": limb_length[3],
        "w0": limb_weight[0],
        "w1": limb_weight[1],
        "w2": limb_weight[2],
        "w3": limb_weight[3],
        "v0": volumes[0],
        "v1": volumes[1],
        "v2": volumes[2],
        "v3": volumes[3],
        "torso_center_height": limb_length[0]/2 + limb_length[1] + limb_length[2] + fixed_distance_to_ground,
    }
    # Take .04f
    for key in data:
        data[key] = int(data[key]*10000) / 10000.
    write_xml(f"./bodies/{mutant_id}.xml", data, body_xml)
    # Don't give the neural network too big an input, the neural network will freak out and output all 1.0's or -1.0's!
    write_yaml(f"./params/{mutant_id}.yaml", data)
    file_list.append(f"./bodies/{mutant_id}.xml")
    param_list.append(f"./params/{mutant_id}.yaml")

config_yaml = {
    "dataset_name": "walker2d",
    "bodies": {
        "total": len(file_list),
        "files": file_list,
        "params": param_list,
    },
    "gym_env": {
        "env_id": "Walker2Ds-v0",
        "filename": "env/walker2d.py",
        "class": "Walker2DEnv",
    }
}
write_yaml("config.yaml", config_yaml)

joint_list = ["thigh_joint", "leg_joint", "foot_joint", "thigh_left_joint", "leg_left_joint", "foot_left_joint"]
foot_list = ["foot", "foot_left"]
power_coef_list = np.array(power_coef_list)
min_power_coef = np.min(power_coef_list, axis=0)
joint_power_coef = []
for idx_joint, idx_power in zip([0,1,2,3,4,5], [1,2,3,1,2,3]):
    joint_power_coef.append(f"self.jdict[\"{joint_list[idx_joint]}\"].power_coef = {min_power_coef[idx_power]}")
joint_power_coef = ";".join(joint_power_coef)
py_data = {
    "action_dim": len(joint_list),
    "obs_dim": 8 + len(joint_list)*2 + len(foot_list),
    "joint_power_coef": joint_power_coef,
    "foot_list_array": foot_list,
}
write_py("env/walker2d.py", py_data, walker2d_py)
