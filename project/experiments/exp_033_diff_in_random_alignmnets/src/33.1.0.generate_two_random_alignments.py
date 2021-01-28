import numpy as np
import hashlib

np.random.seed(0)
order = np.arange(start=0, stop=6)
print(order)

def generate_alignments_1xx():
    orders = []
    for i in range(8):
        _o = np.random.permutation(order)
        _o = ",".join([str(x) for x in _o])
        orders.append(_o)

    orders = "::".join(orders)
    print(orders)
    print(hashlib.md5(orders.encode()).hexdigest())

generate_alignments_1xx()
generate_alignments_1xx()