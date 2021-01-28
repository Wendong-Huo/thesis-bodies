import numpy as np
from common import seeds


class GenerateExp:
    def __init__(self):
        self.alignment_history = {}
        self.selected_bodies = {}

    def construct_random_alignment(self, num_bodies=4, max_joints=8, seed=0):
        orders = []
        with seeds.temp_seed(seed):
            order = np.arange(max_joints)
            for i in range(num_bodies):
                np.random.shuffle(order)
                orders.append(",".join([str(x) for x in order]))
        ret = "::".join(orders)
        return ret

    def check_not_in_alignment_history(self, idx, also_insert=True):
        if idx in self.alignment_history:
            return False
        if also_insert:
            self.alignment_history[idx] = True
        return True

    def check_not_in_body_history(self, idx, also_insert=True):
        if idx in self.selected_bodies:
            return False
        if also_insert:
            self.selected_bodies[idx] = True
        return True

    def _test_repetition(self):
        for i in range(10000000):
            if i % 1000 == 0:
                print(".", end=" ", flush=True)
            alignment = self.construct_custom_alignment(i)
            assert self.check_not_in_alignment_history(alignment)
