import string
import yaml

from termcolor import colored, cprint

g_verbose_level = 2

output_color = {
    -1: ['grey', 'on_red'],  # Error
    0: ['yellow'],           # Warning
    1: ['green'],             # Important info
    2: ['blue'],            # Details
    3: ['white'],            # Debug
}


def output(fstring, verbose_level=3):
    if g_verbose_level >= verbose_level:
        cprint(fstring, *output_color[verbose_level])


def abort(fstring):
    cprint(fstring, *output_color[-1])
    exit(1)


def read_template(filename):
    output(f"Reading template {filename}", 2)
    with open(filename, "r") as f:
        _content = f.read()
    return string.Template(_content)


def read_yaml(filename):
    output(f"Reading yaml {filename}", 2)
    with open(filename, "r") as f:
        return yaml.load(f, Loader=yaml.Loader)
        # return yaml.load(f, Loader=yaml.SafeLoader)


def write_yaml(filename, data):
    output(f"Writing yaml {filename}", 2)
    with open(filename, "w") as f:
        yaml.dump(data, f)


def write_xml(filename, data, body_xml):
    output(f"Writing xml {filename}", 2)
    _content = body_xml.safe_substitute(data)
    with open(filename, "w") as f:
        print(_content, file=f)

def write_script(filename, data, script_template):
    output(f"Writing script {filename}", 2)
    _content = script_template.safe_substitute(data)
    with open(filename, "w") as f:
        print(_content, file=f)

def delete_key(dictionary, key):
    if key in dictionary:
        del dictionary[key]
