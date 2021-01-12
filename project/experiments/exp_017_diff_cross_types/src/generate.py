# read ../input_data/body_templates/*.xml
# generate random variables
# write output_data/bodies_pretrain/
# train all bodies
# select best bodies into output_data/bodies/

from pathlib import Path
import yaml
import string
import math
import numpy as np

body_template_folder = str(Path("../input_data/body-templates").resolve())
body_folder = str(Path("output_data/bodies").resolve())

with open(f"{body_template_folder}/variables.yml", "r") as f:
    variables = yaml.load(f, Loader=yaml.Loader)

def make_body(body, mutate_seed=0):
    with open(f"{body_template_folder}/{body}.xml", "r") as f:
        _content = f.read()
        tempalte_content = string.Template(_content)


    data = variables[f"body-{body}"].copy()

    # body specific calculation
    if body==400:
        data["torso_posterior"] = - data["torso_length"] / 2
        data["torso_anterior"] = data["torso_length"] / 2
        old_data = data.copy()
        for key in old_data:
            if "_angle" in key:
                part_name = key[:-len("_angle")]
                data[f"{part_name}_x"] = data[f"{part_name}_length"] * math.cos(data[f"{part_name}_angle"]/180*math.pi)
                data[f"{part_name}_z"] = data[f"{part_name}_length"] * math.sin(data[f"{part_name}_angle"]/180*math.pi)
        data["neck_x"] = data["torso_anterior"] + data["head_x"]
    elif body==600:
        data["foot_posterior"] = - data["foot_length"] / 3
        data["foot_anterior"] = data["foot_posterior"] + data["foot_length"]
    elif body==800:
        data["torso_pos_arm"] = data["torso_length"] / 2

    # mutate
    if mutate_seed>0:
        largest_noise = 1.4 # Most likely the mutation is within this range (99.7% three deviation)
        np.random.seed(body+mutate_seed)
        mutate_noise = np.random.normal(loc=0, scale=1/3, size=len(data))
        mutate_noise = largest_noise**mutate_noise
        for idx,key in enumerate(data.keys()):
            data[key] *= mutate_noise[idx]
            # print(f"{key}: {data[key]} *= {mutate_noise[idx]}")

    # calculate height
    fixed_height = 0.1
    if body in [300,600,700,800]:
        data["height"] = data["torso_length"] + data["thigh_length"] + data["leg_length"] + data["foot_thickness"] + fixed_height
    elif body==400:
        data["height"] = max(-data["bthigh_z"] - data["bshin_z"] - data["bfoot_z"] + data["bfoot_thickness"], -data["fthigh_z"] - data["fshin_z"] - data["ffoot_z"] + data["ffoot_thickness"]) + fixed_height
    elif body==500:
        data["height"] = data["foot_length"] + fixed_height + 0.2
    
    # keep two decimals
    cleaned_data = {}
    for key in data:
        cleaned_data[key] = int(data[key] * 100)/100

    _content = tempalte_content.safe_substitute(cleaned_data)
    assert "$" not in _content, "some variable not substituted."

    with open(f"{body_folder}/{body+mutate_seed}.xml", "w") as f:
        print(_content, file=f)

for body in [300,400,500,600,700,800]:
    make_body(body)
    for mutation in range(50):
        make_body(body, mutate_seed=mutation)