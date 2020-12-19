import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils

folder = utils.folder

with open(f"{folder}/data-simple.pickle", "rb") as f:
    data = pickle.load(f)

print(data)

def process(_data):
    _data = np.array(_data)
    _two_sigma = 2*np.std(_data)
    _mean = np.mean(_data)
    print(f"{_mean:.0f} +- {_two_sigma:.0f}")
    return _mean, _two_sigma

def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        if int(height)<0:
            va = 'top'
        else:
            va = 'bottom'
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va=va)
error_kw=dict(ecolor=[0.1,0.1,0.1], lw=0.1, capsize=3, capthick=1)

variable_name = "reward"
all_time_max_value = 0
all_time_min_value = 0
fig,ax = plt.subplots(nrows=5, ncols=3, sharey='row', figsize=[8,10])

n_length = 0
reward = data[variable_name]["train_on_one"]
trainon = reward.keys()
for plot_idx, body in enumerate([0,1,2]):
    means = []
    errs = []
    xs = []
    for i in trainon:
        str_prefix = f"train-{body}-"
        if str_prefix in i:
            n_length = len(reward[i])
            m,e = process(reward[i])
            means.append(m)
            errs.append(e)
            xs.append(i[len(str_prefix)+5:])
    # ax[0,body].xaxis.set_tick_params(rotation=45)
    bar = ax[0,plot_idx].bar(xs, means, yerr=errs, color=[0.5]*3, error_kw=error_kw)
    autolabel(bar, ax[0,plot_idx])
    ax[0,plot_idx].set_title(f"Train on {body}")
    max_value = np.max(np.array(means) + np.array(errs))
    min_value = np.min(np.array(means) - np.array(errs))
    all_time_max_value = max_value if all_time_max_value<max_value else all_time_max_value
    all_time_min_value = min_value if all_time_min_value>min_value else all_time_min_value

reward = data[variable_name]["train_on_two"]
trainon = reward.keys()
for plot_idx, body in enumerate([[0,1],[0,2],[1,2]]):
    means = []
    errs = []
    xs = []
    for i in trainon:
        str_prefix = f"train-{body[0]}-{body[1]}-"
        if str_prefix in i:
            n_length = len(reward[i])
            m,e = process(reward[i])
            means.append(m)
            errs.append(e)
            xs.append(i[len(str_prefix)+5:])
    # ax[1,plot_idx].xaxis.set_tick_params(rotation=45)
    bar =ax[1,plot_idx].bar(xs, means, yerr=errs, color=[0.5]*3, error_kw=error_kw)
    autolabel(bar, ax[1,plot_idx])
    ax[1,plot_idx].set_title(f"Train on {body}")
    max_value = np.max(np.array(means) + np.array(errs))
    min_value = np.min(np.array(means) - np.array(errs))
    all_time_max_value = max_value if all_time_max_value<max_value else all_time_max_value
    all_time_min_value = min_value if all_time_min_value>min_value else all_time_min_value

colors_for_columns = [ [0.1, 0.5, 0.7], [0.7, 0.5, 0.1], [0.5, 0.1, 0.7],]
reward = data[variable_name]["train_on_two_with_bodyinfo_zero"]
trainon = reward.keys()
for plot_idx, body in enumerate([[0,1],[0,2],[1,2]]):
    means = []
    errs = []
    xs = []
    for i in trainon:
        str_prefix = f"train-{body[0]}-{body[1]}-"
        if str_prefix in i:
            n_length = len(reward[i])
            m,e = process(reward[i])
            means.append(m)
            errs.append(e)
            xs.append(i[len(str_prefix)+5:])
    # ax[2,plot_idx].xaxis.set_tick_params(rotation=45)
    bar = ax[2,plot_idx].bar(xs, means, yerr=errs, color=colors_for_columns[plot_idx], error_kw=error_kw)
    autolabel(bar, ax[2,plot_idx])
    ax[2,plot_idx].set_title(f"Train on {body} w/ as 0")
    max_value = np.max(np.array(means) + np.array(errs))
    min_value = np.min(np.array(means) - np.array(errs))
    all_time_max_value = max_value if all_time_max_value<max_value else all_time_max_value
    all_time_min_value = min_value if all_time_min_value>min_value else all_time_min_value

reward = data[variable_name]["train_on_two_with_bodyinfo_one"]
trainon = reward.keys()
for plot_idx, body in enumerate([[0,1],[0,2],[1,2]]):
    means = []
    errs = []
    xs = []
    for i in trainon:
        str_prefix = f"train-{body[0]}-{body[1]}-"
        if str_prefix in i:
            n_length = len(reward[i])
            m,e = process(reward[i])
            means.append(m)
            errs.append(e)
            xs.append(i[len(str_prefix)+5:])
    # ax[3,plot_idx].xaxis.set_tick_params(rotation=45)
    bar = ax[3,plot_idx].bar(xs, means, yerr=errs, color=colors_for_columns[plot_idx], error_kw=error_kw)
    autolabel(bar, ax[3,plot_idx])
    ax[3,plot_idx].set_title(f"Train on {body} w/ as 1")
    max_value = np.max(np.array(means) + np.array(errs))
    min_value = np.min(np.array(means) - np.array(errs))
    all_time_max_value = max_value if all_time_max_value<max_value else all_time_max_value
    all_time_min_value = min_value if all_time_min_value>min_value else all_time_min_value


reward = data[variable_name]["train_on_two_with_bodyinfo_two"]
trainon = reward.keys()
for plot_idx, body in enumerate([[0,1],[0,2],[1,2]]):
    means = []
    errs = []
    xs = []
    for i in trainon:
        str_prefix = f"train-{body[0]}-{body[1]}-"
        if str_prefix in i:
            n_length = len(reward[i])
            m,e = process(reward[i])
            means.append(m)
            errs.append(e)
            xs.append(i[len(str_prefix)+5:])
    # ax[4,plot_idx].xaxis.set_tick_params(rotation=45)
    bar = ax[4,plot_idx].bar(xs, means, yerr=errs, color=colors_for_columns[plot_idx], error_kw=error_kw)
    autolabel(bar, ax[4,plot_idx])
    ax[4,plot_idx].set_title(f"Train on {body} w/ as 2")
    max_value = np.max(np.array(means) + np.array(errs))
    min_value = np.min(np.array(means) - np.array(errs))
    all_time_max_value = max_value if all_time_max_value<max_value else all_time_max_value
    all_time_min_value = min_value if all_time_min_value>min_value else all_time_min_value

for i in range(5):
    ax[i,0].set_ylim(all_time_min_value*1.05, all_time_max_value*1.05) # adjust ylim for all subplots

fig.suptitle(f"Cross Testing {variable_name} (N={n_length} for each test)", fontsize=16)
plt.tight_layout()
plt.savefig(f"{folder}/cross_test_{variable_name}.png")
# plt.show()
