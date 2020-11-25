import matplotlib.pyplot as plt
import numpy as np

"""
        * body: 10, variation: 10% => 33.7 +- 4.4 vs 31.7 +- 2.2 (body:i3_s0,i7_s0,i7_s1 is missing; nobody:i1_s0, i1_s1, i2_s0, i3_s1, i4_s1, i6_s0, i8_s1 is missing.)

        * body: 20, variation: 10% => 35.0 +- 4.2 vs 33.4 +- 4.2

        * body: 50, variation: 10% => 26.2 +- 5.1 vs 28.1 +- 6.0

        * body: 100, variation: 10% => 9.6 +- 10.2 vs 8.5 +- 9.3 (stop earlier at 4e6, not 5e6)

        * body: 20, variation: 20% => 33.0 +- 6.9 vs 28.8 +- 5.0

        * body: 20, variation: 50% => 7.6 +- 6.1 vs 8.8 +- 6.9

        * body: 20, variation: 90% => 1.5 +- 0.8 vs 1.6 +- 1.0
"""


fig, axes = plt.subplots(nrows=2)
ax = axes[0]
x = [10, 20, 50, 100]
y_body = np.array([33.7, 35.0, 26.2, 9.6])
y_std_body = np.array([4.4, 4.2, 5.1, 10.2])

y_nobody = np.array([31.7, 33.4, 28.1, 8.5])
y_std_nobody = np.array([2.2, 4.2, 6.0, 9.3])

ax.set_title(r"Variation 10%")
ax.plot(x, y_body, "o-", c="#3333FF", label="body")
ax.plot(x, y_nobody, "o-", c="#FF3333", label="no_body")
ax.fill_between(x, y_body-y_std_body, y_body+y_std_body, color="#3333FF", alpha=0.1)
ax.fill_between(x, y_nobody-y_std_nobody, y_nobody+y_std_nobody, color="#FF3333", alpha=0.1)
ax.set_xlabel("Num of Bodies in dataset")
ax.set_ylabel("Distance")
ax.legend()

ax = axes[1]
x = [10, 20, 50, 90]
y_body = np.array([35.0, 33.0, 7.6, 1.5])
y_std_body = np.array([4.2, 6.9, 6.1, 0.8])

y_nobody = np.array([33.4, 28.8, 8.8, 1.6])
y_std_nobody = np.array([4.2, 5.0, 6.9, 1.0])

ax.set_title(r"Num of Bodies 20")
ax.plot(x, y_body, "o-", c="#3333FF", label="body")
ax.plot(x, y_nobody, "o-", c="#FF3333", label="no_body")
ax.fill_between(x, y_body-y_std_body, y_body+y_std_body, color="#3333FF", alpha=0.1)
ax.fill_between(x, y_nobody-y_std_nobody, y_nobody+y_std_nobody, color="#FF3333", alpha=0.1)
ax.set_xlabel("Num of Bodies in dataset")
ax.set_ylabel("Distance")
ax.legend()

plt.tight_layout()
plt.savefig("failed_experiment.png")