import pickle
import hashlib

with open("output_data/jobs_vanilla4.pickle", "rb") as f:
    data = pickle.load(f)
# print(data)

to_find = "89bd5344c0c00760f89c4a39af49de35"
for i in data:
    str2hash = data[i]["custom_alignment"]
    ret = hashlib.md5(str2hash.encode()).hexdigest()
    # print(ret)
    if to_find==ret:
        print(data[i])
