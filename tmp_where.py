import numpy as np

a = np.zeros([3,3,3])
a[0,0,0] = 100
a[0,0,1] = 100
b = np.where(a>0)
print(a[b])
