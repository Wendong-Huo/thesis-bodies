import pickle
import hashlib

with open("output_data/jobs_random_exp.pickle", "rb") as f:
    data = pickle.load(f)
# print(data)

to_find = "5532bcd5b1427a7cbcecb2425c7cd702"
for i in data:
    str2hash = data[i]["custom_alignment"]
    ret = hashlib.md5(str2hash.encode()).hexdigest()
    # print(ret)
    if to_find==ret:
        print(data[i])

import brotli
l = "0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7::0,1,2,3,4,5,6,7"
compressed_data = brotli.compress(l.encode())
print(compressed_data)
data = brotli.decompress(compressed_data)
print(data)
