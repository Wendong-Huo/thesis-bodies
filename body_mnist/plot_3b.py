import matplotlib.pyplot as plt
import glob,os
import yaml
import numpy as np
from yaml.loader import SafeLoader

def process(_data):
    _data = np.array(_data)
    _n = _data.shape[0]
    _two_sigma = 2*np.std(_data)
    _mean = np.mean(_data)
    print(f"{_mean:.0f} +- {_two_sigma:.0f} (n={_n})")
    return _mean, _two_sigma
error_kw=dict(ecolor=[0.1,0.1,0.1], lw=0.1, capsize=3, capthick=1)

with open("exp_3b/total_jobs.yml") as f:
    total_jobs = yaml.load(f, Loader=SafeLoader)

rs = []
es = []
r1s = []
e1s = []
ticks = []
do_break = False
for i, job in enumerate(total_jobs):
    if do_break:
        break
    # print(total_jobs[i]['test'])
    # print(total_jobs[i]['train'])
    str_train = '-'.join(str(x) for x in total_jobs[i]['train'])
    # print(str_train)
    for test in total_jobs[i]['test']:
        _class = test//100
        fname = f"exp_3b/test-results/model-ant-{str_train}-test-{test}-class-0.yaml"
        fname_with_bodyinfo = f"exp_3b/test-results/model-ant-{str_train}-with-bodyinfo-test-{test}-class-{_class}.yaml"
        # print(fname)
        if not os.path.exists(fname):
            print(os.path.exists(fname))
        # print(fname_with_bodyinfo)
        if not os.path.exists(fname_with_bodyinfo):
            print(os.path.exists(fname_with_bodyinfo))

        with open(fname, "r") as f:
            result = yaml.load(f, Loader=yaml.SafeLoader)
        r,e = process(result["total_reward"])
        rs.append(r); es.append(e)
        with open(fname_with_bodyinfo, "r") as f:
            result = yaml.load(f, Loader=yaml.SafeLoader)
        r1,e1 = process(result["total_reward"])
        r1s.append(r1); e1s.append(e1)
        ticks.append(fname)

        if r>r1 * 1.5:
            print(fname)
            do_break = True

fig,ax = plt.subplots()
ind = np.arange(len(rs))
width = 0.35
ax.bar(ind, rs, width, color='blue', yerr=es, error_kw=error_kw)
ax.bar(ind+width, r1s, width, color='green', yerr=e1s, error_kw=error_kw)
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(ticks, rotation = 90, ha="right")
plt.subplots_adjust(bottom=0.6)
plt.show()

exit(0)

filenames = glob.glob("exp_3b/test-results/model-ant-*.yaml")

total_rewards = []
total_rewards_with_bodyinfo = []
for filename in filenames:
    with open(filename, "r") as f:
        result = yaml.load(f, Loader=yaml.SafeLoader)
    assert isinstance(result["total_reward"], list), "results should be a list, not np.array"
    print(filename)
    process(result["total_reward"])
    if "with-bodyinfo" in filename:
        total_rewards_with_bodyinfo += result["total_reward"] 
    else:
        total_rewards += result["total_reward"] 

mean, err = process(total_rewards)
mean_with_bodyinfo, err_with_bodyinfo = process(total_rewards_with_bodyinfo)

print(mean,err)
print(mean_with_bodyinfo,err_with_bodyinfo)