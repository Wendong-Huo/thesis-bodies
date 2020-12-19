import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils

folder = utils.folder

with open(f"{folder}/test-results.pickle", "rb") as f:
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

colors_for_columns = [ [0.1, 0.5, 0.7], [0.7, 0.5, 0.1], [0.5, 0.1, 0.7],]

variable_name = "reward"
all_time_max_value = 0
all_time_min_value = 0
fig,ax = plt.subplots(nrows=6, ncols=3, sharey='row', figsize=[8,10])
data = data[variable_name]
# Train on one body (Standard RL)
gs = ax[0,0].get_gridspec()
# remove the underlying axes
for _ax in ax[0, :]:
    _ax.remove()
axbig = fig.add_subplot(gs[0, :])

xs = []
means = []
errs = []
for str_key in data[0]:
    key = [int(x) for x in str_key.split('-')]
    if len(key) == 1:
        print(key)
        xs.append(key[0])
        rets = data[0][str_key][key[0]][0]
        n_length = len(rets)
        mean, err = process(rets)
        means.append(mean)
        errs.append(err)
        print(mean, err)
xs, means, errs = zip(*sorted(zip(xs,means, errs)))
xs = [str(x) for x in xs]
bar = axbig.bar(xs, means, yerr=errs, color=[0.5]*3, error_kw=error_kw)
autolabel(bar, axbig)
max_value = np.max(np.array(means) + np.array(errs))
min_value = np.min(np.array(means) - np.array(errs))
all_time_max_value = max_value if all_time_max_value<max_value else all_time_max_value
all_time_min_value = min_value if all_time_min_value>min_value else all_time_min_value


# Train on bodies from the same group (Standard RL)
# train on
trainons = []
for str_key in data[0]:
    key = [int(x) for x in str_key.split('-')]
    if len(key) == 4:
        trainons.append(str_key)
trainons.sort()
col_id = -1
for str_key in trainons:
    xs = []
    means = []
    errs = []
    key = [int(x) for x in str_key.split('-')]
    if len(key) == 4:
        col_id += 1
        print(key)
        testons = list(data[0][str_key].keys())
        for teston in testons:
            xs.append(teston)
            rets = data[0][str_key][teston][0]
            n_length = len(rets)
            mean, err = process(rets)
            means.append(mean)
            errs.append(err)
            print(mean, err)
        xs, means, errs = zip(*sorted(zip(xs,means, errs)))
        xs = [str(x) for x in xs]
        bar = ax[1 + col_id//3, col_id%3].bar(xs, means, yerr=errs, color=[0.5]*3, error_kw=error_kw)
        autolabel(bar, ax[1 + col_id//3, col_id%3])
        ax[1 + col_id//3, col_id%3].set_title(str_key)
        max_value = np.max(np.array(means) + np.array(errs))
        min_value = np.min(np.array(means) - np.array(errs))
        all_time_max_value = max_value if all_time_max_value<max_value else all_time_max_value
        all_time_min_value = min_value if all_time_min_value>min_value else all_time_min_value

# Train on bodies from the two groups (Standard RL)
trainons = []
for str_key in data[0]:
    key = [int(x) for x in str_key.split('-')]
    if len(key) == 8:
        trainons.append(str_key)
trainons.sort()
col_id = -1
for str_key in trainons:
    xs = []
    means = []
    errs = []
    key = [int(x) for x in str_key.split('-')]
    if len(key) == 8:
        col_id += 1
        print(key)
        testons = list(data[0][str_key].keys())
        for teston in testons:
            xs.append(teston)
            rets = data[0][str_key][teston][0]
            n_length = len(rets)
            mean, err = process(rets)
            means.append(mean)
            errs.append(err)
            print(mean, err)
        xs, means, errs = zip(*sorted(zip(xs,means, errs)))
        xs = [str(x) for x in xs]
        bar = ax[3,col_id].bar(xs, means, yerr=errs, color=colors_for_columns[col_id], error_kw=error_kw)
        autolabel(bar, ax[3,col_id])
        ax[3, col_id].set_title(str_key)
        max_value = np.max(np.array(means) + np.array(errs))
        min_value = np.min(np.array(means) - np.array(errs))
        all_time_max_value = max_value if all_time_max_value<max_value else all_time_max_value
        all_time_min_value = min_value if all_time_min_value>min_value else all_time_min_value

# Train on bodies from the two groups (with bodyinfo, as group 0)
for group in [0,1]:
    trainons = []
    for str_key in data[0]:
        key = [int(x) for x in str_key.split('-')]
        if len(key) == 8:
            trainons.append(str_key)
    trainons.sort()
    col_id = -1
    for str_key in trainons:
        xs = []
        means = []
        errs = []
        key = [int(x) for x in str_key.split('-')]
        if len(key) == 8:
            col_id += 1
            print(key)
            testons = list(data[0][str_key].keys())
            for teston in testons:
                xs.append(teston)
                rets = data[1][str_key][teston][group]
                n_length = len(rets)
                mean, err = process(rets)
                means.append(mean)
                errs.append(err)
                print(mean, err)
            xs, means, errs = zip(*sorted(zip(xs,means, errs)))
            xs = [str(x) for x in xs]
            bar = ax[4+group,col_id].bar(xs, means, yerr=errs, color=colors_for_columns[col_id], error_kw=error_kw)
            autolabel(bar, ax[4+group,col_id])
            ax[4+group, col_id].set_title(str_key)
            max_value = np.max(np.array(means) + np.array(errs))
            min_value = np.min(np.array(means) - np.array(errs))
            all_time_max_value = max_value if all_time_max_value<max_value else all_time_max_value
            all_time_min_value = min_value if all_time_min_value>min_value else all_time_min_value


for i in range(ax.shape[0]):
    ax[i,0].set_ylim(all_time_min_value*1.05, all_time_max_value*1.05) # adjust ylim for all subplots

fig.suptitle(f"Cross Testing {variable_name} (N={n_length} for each test)", fontsize=16)
plt.tight_layout()
plt.savefig(f"{folder}/cross_test_{variable_name}.png")
# plt.show()
