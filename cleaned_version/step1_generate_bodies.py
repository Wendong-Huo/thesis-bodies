import os
import shutil
import random

from arguments import get_args
from utils import output, abort, read_template, read_yaml, write_xml, write_yaml

args = get_args()


def generate_bodies():
    random.seed(args.seed_bodies)
    output(f"Start generating bodies {args.num_bodies}", 1)

    # 1. Check templates
    template_files = check_templates()

    # 2. Create Folders
    dataset_path = create_folder()

    # 3. Generate variations and write body, param files
    body_xml = read_template(template_files[0])
    body_yaml = read_yaml(template_files[1])
    file_list, param_list = [], []
    for i in range(args.num_bodies):
        data = {}
        for key in body_yaml['variable']:
            data[key] = body_yaml['variable'][key] * ((random.random() * 2 - 1) * args.body_variation_range / 100 + 1.0)
            data[key] = significant_digits(data[key], 4)
        for key in body_yaml['fixed']:
            data[key] = body_yaml['fixed'][key]
        for key in body_yaml['combination']:
            data[key] = 0
            for key1 in body_yaml['combination'][key]:
                data[key] += data[key1]
            data[key] = significant_digits(data[key], 4)
        # Volume calculation
        for part in body_yaml['part']:
            data[f"volume_{part}"] = data[f"length_{part}"] * 3.14 * data[f"weight_{part}"] * data[f"weight_{part}"]

        write_xml(f"{dataset_path}/bodies/{i}.xml", data, body_xml)
        write_yaml(f"{dataset_path}/params/{i}.yaml", data)
        file_list.append(f"bodies/{i}.xml")
        param_list.append(f"params/{i}.yaml")

    # 4. Write config file
    env_id = f"{args.template_body}_{args.num_bodies}_{args.body_variation_range}-v{args.seed_bodies}"
    config_yaml = {
        "dataset_name": args.template_body,
        "bodies": {
            "total": len(file_list),
            "files": file_list,
            "params": param_list,
        },
        "gym_env": {
            "env_id": env_id,
            "filename": f"{args.template_body}.py",
            "class": f"{args.template_body.capitalize()}Env",
        }
    }
    write_yaml(f"{dataset_path}/config.yaml", config_yaml)

    # 5. Copy over Gym Env Python file
    shutil.copyfile(template_files[2], f"{dataset_path}/{args.template_body}.py")

    return env_id, dataset_path


def check_templates():
    template_files = [None]*3
    template_files[0] = f"body-templates/{args.template_body}.template.xml"
    template_files[1] = f"body-templates/{args.template_body}.yaml"
    template_files[2] = f"body-templates/{args.template_body}.py"

    for f in template_files:
        if not os.path.exists(f):
            abort(f"Template file not found: {f}")
    return template_files


def significant_digits(var, num_digit):
    n = 10**num_digit
    return int(var*n)/n


def create_folder():
    dataset_path = f"dataset/{args.template_body}_{args.num_bodies}_{args.body_variation_range}-v{args.seed_bodies}"
    if os.path.exists(dataset_path):
        if args.dataset_exist_ok:
            output(f"Overwrite dataset {args.template_body}.", 0)
            shutil.rmtree(dataset_path)
            os.mkdir(dataset_path)
        else:
            abort(f"Dataset exists: {dataset_path}")
    else:
        os.mkdir(dataset_path)
    os.mkdir(f"{dataset_path}/bodies")
    os.mkdir(f"{dataset_path}/params")
    return dataset_path


if __name__ == "__main__":
    generate_bodies()
