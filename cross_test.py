import pickle
import numpy as np
from stable_baselines3.common.logger import record
import enjoy
train_body_id = 0
test_body_id = 0

def all():
    records_xy = []
    for train_body_id in range(100):
        print(f" Body trained on {train_body_id}")
        print("")
        records = []
        for test_body_id in range(100):
            record = enjoy.enjoy(
                stats_path=f"logs_3x100/{train_body_id}_0/ppo/Walker2Ds-v0_1/Walker2Ds-v0", 
                model_path=f"logs_3x100/{train_body_id}_0/ppo/Walker2Ds-v0_1/best_model.zip", 
                dataset="dataset/walker2d_v6",
                body_id=test_body_id, 
                n_timesteps=1000, test_time=3, render=False)
            records.append(record)
            print(f"test on {test_body_id}")
            print(f"distances: {record}")
        records_xy.append(records)
    return np.array(records_xy)

def one(train_body_id, run, test_body_id):
    record = enjoy.enjoy(
        stats_path=f"logs_3x100/{train_body_id}_{run}/ppo/Walker2Ds-v0_1/Walker2Ds-v0", 
        model_path=f"logs_3x100/{train_body_id}_{run}/ppo/Walker2Ds-v0_1/best_model.zip", 
        dataset="dataset/walker2d_v6",
        body_id=test_body_id, 
        n_timesteps=1000, test_time=3, render=True)
    # mean_record = np.mean(record)
    print(f"test on {test_body_id}")
    print(f"distances: {record}")

one(9, 0, 9)
if False:
    try:
        raise "a"
        with open("tmp/cross_test_records.pickle", "rb") as f:
            records = pickle.load(f)
    except:
        records = all()
        with open("tmp/cross_test_records.pickle", "wb") as f:
            pickle.dump(records,f)

    # print(records)
    import matplotlib.pyplot as plt
    import scipy.stats

    CIs = []
    for i in range(100):
        sample = records[i]
        confidence_level = 0.9
        degrees_freedom = sample.size - 1
        sample_mean = np.mean(sample)
        sample_standard_error = scipy.stats.sem(sample)
        confidence_interval = scipy.stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
        CIs.append(confidence_interval)
        print(f"{i}: {confidence_interval}")
    CIs = np.array(CIs)
    mean_record = np.mean(records, axis=1)
    ranks = np.argsort(mean_record)[::-1]
    records_max = np.max(records, axis=1)
    records_min = np.min(records, axis=1)
    xs = np.arange(0,100)
    CIs = np.swapaxes(CIs, 0, 1)
    plt.bar(xs, mean_record[ranks], alpha=.7)
    CIs = CIs[:,ranks]
    plt.fill_between(xs, records_min[ranks], records_max[ranks], color='b', alpha=.2)
    plt.xticks([])
    plt.xlabel("test bodies, order by score desc.")
    plt.ylabel("test score")
    plt.show()
