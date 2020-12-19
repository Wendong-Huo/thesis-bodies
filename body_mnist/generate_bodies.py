import os
import shutil
import random
import math
import numpy as np

import utils
from utils import output, abort, read_template, read_yaml, write_xml, write_yaml

def generate(class_type="classA", id_base=0):
    os.makedirs(f"{utils.folder}/envs", exist_ok=True)
    xml = read_template("body-templates/ant.template.xml")
    conf = read_yaml("body-templates/ant.yaml")
    data = conf[class_type]
    write_xml(f"{utils.folder}/envs/{id_base}.mean.xml", data, xml)

    for i in range(5):
        data_clone = data.copy()
        for key in data_clone:
            data_clone[key] += np.random.normal(0, 0.01)
        write_xml(f"{utils.folder}/envs/{i+id_base}.xml", data_clone, xml)

if __name__ == "__main__":
    np.random.seed(0)
    generate("classA", id_base=0)
    generate("classB", id_base=100)

