from common import seeds
import numpy as np

def construct_custom_alignment(seed, num_joints, num_bodies, num_envs):
    assert num_envs%num_bodies==0
    repeat_times = num_envs//num_bodies
    orders = []
    with seeds.temp_seed(seed):
        order = np.arange(num_joints)
        for i in range(num_bodies):
            np.random.shuffle(order)
            orders.append(",".join([str(x) for x in order]))
    unique_alignment = "::".join(orders)
    repeated_alignments = []
    for i in range(repeat_times):
        repeated_alignments.append(unique_alignment)
    ret = "::".join(repeated_alignments)
    return ret

def get_alignments_to_test(seed=0, num_tests=10, num_joints=8, num_bodies=8, num_envs=16):
    alignments_to_test = []
    for i in range(num_tests):
        alignment = construct_custom_alignment(seed=seed*num_tests+i, num_joints=num_joints, num_bodies=num_bodies, num_envs=num_envs)
        alignments_to_test.append(alignment)

    return alignments_to_test

alignment_history = {}
def check_not_in_alignment_history(idx, also_insert=True):
    global alignment_history
    if idx in alignment_history:
        return False
    if also_insert:
        alignment_history[idx] = True
    return True

def test_repetition():
    for i in range(10000000):
        if i % 1000 == 0:
            print(".", end=" ", flush=True)
        alignment = construct_custom_alignment(i,8,8,8)
        assert check_not_in_alignment_history(alignment)

# test_repetition()
