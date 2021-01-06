import numpy as np

a = np.array([17,22,4,2,14,24,0,5,8,7,27,28,15,25,12,9,20,3,29,16,10,6,1,13,18,23,11,26,21,19])
np.random.shuffle(a)
print(a.tolist())
np.random.shuffle(a)
print(a.tolist())

np.random.shuffle(a)
print(a.tolist())
