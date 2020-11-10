import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

with open("read_tb.pickle", "rb") as f:
    (max_body_xss, max_body_x_stepss) = pickle.load(f)
mean_xs = np.mean(max_body_xss, axis=1)
arg = np.argsort(mean_xs)[::-1]

all_data = []
for i in range(100):
    with open(f"misc/all/train_on_{i}.pickle", "rb") as f:
        ret = pickle.load(f)
    print(ret.shape)
    all_data.append(ret)
all_data = np.array(all_data)
# all_data = all_data[:,:,:,2] # only choose 1/3 data
all_data = all_data.reshape([100,100,-1])
all_data = np.mean(all_data, axis=2)
# np.random.seed(0)
# data = np.random.random([100,100,3,3])

# data = data.reshape([100,100,-1])
# data = np.mean(data, axis=2)
# for i in range(100):
#     data[i,i] = mean_xs[i]


# for i in range(100):
#     print(f"check {all_data[i,i]} == {mean_xs[i]}")

all_data = all_data[arg,:]
all_data = all_data[:,arg]
# all_data[72,26] = 45
plt.figure(figsize=(14.0, 10.0))
ax = plt.gca()
im = ax.matshow(all_data.T)
ax.set_xlabel("Trained on bodies, order by same-body performance")
ax.set_ylabel("Tested on bodies, order by same-body performance")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.tight_layout()
# plt.show()
plt.savefig("cross_test.png")
plt.close()
