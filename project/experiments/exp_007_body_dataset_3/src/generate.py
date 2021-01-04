# read ../input_data/body_templates/*.xml
# generate random variables
# write output_data/bodies_pretrain/
# train all bodies
# select best bodies into output_data/bodies/

from pathlib import Path
import yaml
import string
import math

body_template_folder = str(Path("../input_data/body-templates").resolve())
body_folder = str(Path("output_data/bodies").resolve())

with open(f"{body_template_folder}/variables.yml", "r") as f:
    variables = yaml.load(f, Loader=yaml.Loader)

def make_body(body, fatter):
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

    # make robot fatter
    if fatter:
        for key in data:
            if "_thickness" in key:
                data[key] += .05
    # keep two decimals
    cleaned_data = {}
    for key in data:
        cleaned_data[key] = int(data[key] * 100)/100

    _content = tempalte_content.safe_substitute(cleaned_data)
    assert "$" not in _content, "some variable not substituted."

    with open(f"{body_folder}/{body+(1 if fatter else 0)}.xml", "w") as f:
        print(_content, file=f)

for body in [300,400,500,600]:
    make_body(body, fatter=True)
    make_body(body, fatter=False)