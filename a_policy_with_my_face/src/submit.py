import os
from time import sleep

bodies = [10,20,50,100]
variations = [10,20,50,90]
for b in bodies:
    for v in variations:
        cmd = f"python main.py --no-single --num-bodies={b} --body-variation-range={v} --seed-bodies=50 --vacc -p {'short' if b<50 else 'bluemoon'} --n-timesteps=5e6"
        print(cmd)
        # os.system(cmd)
        # sleep(5)
