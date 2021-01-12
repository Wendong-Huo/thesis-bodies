# import numpy as np

# a = np.array([17,22,4,2,14,24,0,5,8,7,27,28,15,25,12,9,20,3,29,16,10,6,1,13,18,23,11,26,21,19])
# np.random.shuffle(a)
# print(a.tolist())
# np.random.shuffle(a)
# print(a.tolist())

# np.random.shuffle(a)
# print(a.tolist())
need_rerun = [
    {'bodies': [300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315],
     'seed': 4, 'method': 'align'},
    {'bodies': [300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315],
     'seed': 6, 'method': 'random'},
    {'bodies': [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515],
     'seed': 0, 'method': 'align'},
    {'bodies': [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515],
     'seed': 4, 'method': 'align'},
    {'bodies': [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515],
     'seed': 6, 'method': 'align'},
    {'bodies': [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515],
     'seed': 0, 'method': 'random'},
    {'bodies': [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515],
     'seed': 3, 'method': 'random'},
    {'bodies': [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515],
     'seed': 5, 'method': 'random'},
    {'bodies': [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515],
     'seed': 6, 'method': 'random'},
    {'bodies': [600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615],
     'seed': 4, 'method': 'align'},
    {'bodies': [600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615],
     'seed': 6, 'method': 'align'}]

print("\n"*2)
for r in need_rerun:
    cmd = f"sbatch -J exp_010_rerun submit.sh python 1.train.py --seed={r['seed']} --train_bodies={','.join([str(x) for x in r['bodies']])} --test_bodies={','.join([str(x) for x in r['bodies']])}"
    if r["method"]=='random':
        cmd += " --random_align_obs"
    

    print(cmd)
print("\n"*2)
