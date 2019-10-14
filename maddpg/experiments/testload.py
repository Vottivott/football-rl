import pickle
import os
import time
from os import listdir
from os.path import isfile, join

while 1:
    path = "../../worker_experiences/"
    t0 = time.time()
    experience_files = [f for f in listdir(path) if isfile(join(path, f))]
    num_files = len(experience_files)
    for fname in experience_files:
        fpath = join(path, fname)
        with open(fpath, "rb") as f:
            d = pickle.load(f)
            print(len(d))
            print(len(d[0]))
            print(d[0])
        if False:
            os.remove(fpath)
    print("Loaded %d games from %d files (in %.2f seconds)" % (200*num_files, num_files, time.time()-t0))
    time.sleep(3)
