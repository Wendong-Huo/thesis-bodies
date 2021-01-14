import numpy as np

fn = np.random.choice(range(5), replace=False, size=[5])
print(fn)
antifn = np.argsort(fn)
print(antifn)

input_data = np.random.choice(range(5), replace=False, size=[5])
print(input_data)
data = input_data[fn]

output_data = data[antifn]
print(output_data)