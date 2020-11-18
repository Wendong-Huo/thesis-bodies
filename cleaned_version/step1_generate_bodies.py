import os
import shutil
import string
import yaml

from arguments import args
from utils import output, abort


def generate_bodies():
    output(f"Start generating bodies {args.num_bodies}", 0)
    dataset_path = create_folder()
    template_files = [None]*3
    template_files[0] = f"body-templates/{args.template_body}.template.xml"
    template_files[1] = f"body-templates/{args.template_body}.yaml"
    template_files[2] = f"body-templates/{args.template_body}.py"

    for f in template_files:
        if not os.path.exists(f):
            abort(f"Template file not found: {f}")

    body_xml = read_template(template_files[0])
    body_yaml = read_yaml(template_files[1])

    output(body_xml, 2)
    output(body_yaml, 2)

def create_folder():
    dataset_path = f"dataset/{args.name_bodies}"
    if os.path.exists(dataset_path):
        if args.name_bodies_exist_ok:
            output(f"Overwrite dataset {args.name_bodies}.", 1)
            shutil.rmtree(dataset_path)
            os.mkdir(dataset_path)
        else:
            abort(f"Dataset exists: {dataset_path}")
    else:
        os.mkdir(dataset_path)
    os.mkdir(f"{dataset_path}/bodies")
    os.mkdir(f"{dataset_path}/params")
    return dataset_path

def read_template(filename):
    with open(filename, "r") as f:
        _content = f.read()
    return string.Template(_content)

def read_yaml(filename):
    with open(filename, "r") as f:
        return yaml.load(f)

if __name__ == "__main__":
    generate_bodies()
