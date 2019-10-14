import pickle
import os
import time
from os import listdir
from os.path import isfile, join

def load_new_experiences():
    path = "../../worker_experiences/"
    t0 = time.time()
    experience_files = sorted([f for f in listdir(path) if isfile(join(path, f))])
    num_files = len(experience_files)
    experiences = []
    for fname in experience_files:
        fpath = join(path, fname)
        try:
            with open(fpath, "rb") as f:
                experiences.extend(pickle.load(f))
            if True:
                os.remove(fpath)
        except OSError:
            print("OSError in load_new_experiences")
        except EOFError:
            print("EOFError in load_new_experiences - skipping 1 file")
        except Exception as e:
            print("Error occured. (Arguments {0}) - skipping 1 file".format(e.args))
    print("Loaded %d games from %d files (in %.2f seconds)" % (200 * num_files, num_files, time.time() - t0))
    return experiences