import common.arguments as arguments
from common.utils import *
from common.robo_common import *

check_exp_folder()

args = arguments.get_args()
seed = args.seed

output_data_folder = get_output_data_folder()
input_data_folder = get_input_data_folder()
current_folder = get_current_folder()