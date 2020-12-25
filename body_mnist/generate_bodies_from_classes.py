import os
import shutil
import random
import math
import numpy as np

import utils
from utils import output, abort, read_template, read_yaml, write_xml, write_yaml

def generate(class_type="classA", id_base=0):
    os.makedirs(f"{utils.folder}/envs", exist_ok=True)
    os.makedirs(f"{utils.folder}/params", exist_ok=True)
    xml = read_template("body-templates/ant.template.xml")
    conf = read_yaml("body-templates/ant.yaml")
    data = conf[class_type]
    write_xml(f"{utils.folder}/envs/{id_base}.mean.xml", data, xml)

    for i in range(5):
        data_clone = data.copy()
        for key in data_clone:
            data_clone[key] += np.random.normal(0, 0.01)
        
        data_clone["_initial_z"] = data_clone["size_torso"] + 0.5
        write_xml(f"{utils.folder}/envs/{i+id_base}.xml", data_clone, xml)
        params = {
            "size_torso": data_clone["size_torso"]
        }
        write_yaml(f"{utils.folder}/params/{i+id_base}.yml", params)

if __name__ == "__main__":
    np.random.seed(0)
    generate("class_103", id_base=0)
    # generate("class_169", id_base=100)
    generate("class_238", id_base=100)
