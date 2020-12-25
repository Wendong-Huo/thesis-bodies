import numpy as np
import utils
from utils import output, abort, read_template, read_yaml, write_xml, write_yaml

np.random.seed(0)

def generate(N_bodies=500):
    id_base = 0
    xml = read_template("body-templates/ant.template.xml")
    conf = read_yaml("body-templates/ant.yaml")
    data = conf["classA"]
    write_xml(f"{utils.folder}/envs/{id_base}.mean.xml", data, xml)

    for i in range(N_bodies):
        data_clone = data.copy()
        for key in data_clone:
            data_clone[key] += np.random.normal(0, 0.1)
        write_xml(f"{utils.folder}/envs/{i+id_base}.xml", data_clone, xml)

if __name__ == "__main__":
    generate()