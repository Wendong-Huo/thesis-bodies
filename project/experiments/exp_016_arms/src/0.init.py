import common.common as common

common.get_output_data_folder(init=True)
settings = """
{
    "python.analysis.extraPaths": ["./experiments/"""+common.get_exp_name()+"""/src"]
}
"""

# print(settings)
with open("../../../.vscode/settings.json", "w") as f:
    print(settings, file=f)